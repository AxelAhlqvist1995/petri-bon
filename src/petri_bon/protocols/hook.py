"""The BoN auditor-generate hook.

``bon_generate`` returns an ``AuditorGenerate`` callable for
``inspect_petri.auditor_agent(generate=...)``. Petri calls it once per
auditor turn; every intermediate generation (drafts, refinements, scoring,
feedback) happens inside the hook and never enters the auditor's message
history — only the returned winner does. This gives the critique-refine
"context trim" property for free: the next turn starts with a clean auditor
context containing no trace of the loop.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Literal

from inspect_ai.log import transcript
from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageTool,
    Model,
    ModelOutput,
)
from inspect_ai.tool import Tool
from inspect_ai.util import store

from ..feedback import DefaultFeedbackModel, FeedbackProvider, build_refine_message
from ..rendering import (
    describe_action_sequence,
    render_target_transcript,
    target_visible_calls,
)
from ..selectors.base import CandidateScorer, ScoreSelector, Selector
from ..types import (
    UNSCORABLE_SENTINEL,
    Candidate,
    TurnContext,
    format_preference_band,
)
from ..validation import validate_candidate

logger = logging.getLogger(__name__)

STORE_KEY = "petri_bon:turns"


def bon_generate(
    *,
    selector: Selector | CandidateScorer,
    feedback: FeedbackProvider | None = None,
    protocol: Literal["parallel", "critique_refine"] = "critique_refine",
    n_samples: int = 1,
    rounds: int = 2,
    score_original: bool = True,
    select_from_all: bool = True,
    score_formatter=None,
    refine_template_base: str = "preference",
):
    """Build an ``AuditorGenerate`` hook running a BoN protocol per turn.

    Args:
        selector: A ``Selector`` (picks the winner directly, e.g. pairwise) or
            a ``CandidateScorer`` (pointwise; wrapped in ``ScoreSelector``).
        feedback: Critique provider for refinement rounds. Defaults to
            ``DefaultFeedbackModel()`` when the protocol needs one.
        protocol: ``"parallel"`` (sample n, pick best — no refinement) or
            ``"critique_refine"`` (iteratively critique and refine each chain).
        n_samples: Candidates per round (independent refinement chains).
        rounds: Refinement rounds for ``critique_refine``.
        score_original: Include round-0 drafts in the selection pool.
        select_from_all: Select across all rounds, not just the final one.
        score_formatter: How scores render in prompts (banded by default).
        refine_template_base: Auditor-feedback template family
            (``"preference"`` or ``"pairwise"``).
    """
    scorer: CandidateScorer | None
    if isinstance(selector, Selector):
        sel: Selector = selector
        scorer = getattr(selector, "scorer", None)
    else:
        scorer = selector
        sel = ScoreSelector(scorer)

    if protocol == "parallel":
        rounds = 0
        score_original = True
    if rounds > 0 and feedback is None:
        feedback = DefaultFeedbackModel(score_formatter=score_formatter)

    fmt = score_formatter or format_preference_band
    turn_counter = {"turn": 0}

    async def generate(
        model: Model,
        messages: list[ChatMessage],
        tools: list[Tool],
        cache: bool | CachePolicy,
    ) -> ModelOutput:
        from inspect_petri.target import controller

        turn_counter["turn"] += 1
        turn = turn_counter["turn"]
        state = controller().state
        ctx = TurnContext(
            turn=turn,
            seed_instructions=state.seed_instructions or "",
            auditor_messages=list(messages),
            target_messages=list(state.messages),
            target_tools=list(state.tools),
            tools=tools,
            cache=cache,
            auditor_model=model,
        )

        async def draft(index: int) -> Candidate:
            output = await model.generate(
                input=list(messages), tools=tools, cache=cache
            )
            return Candidate(message=output.message, output=output, round=0, index=index)

        async def prepare(cand: Candidate) -> Candidate:
            """Validate and (when a pointwise scorer exists) score a candidate."""
            ok, err = validate_candidate(cand.message, ctx.target_messages)
            if not ok:
                cand.structural_error = err
                cand.score = UNSCORABLE_SENTINEL
                return cand
            cand.details["transcript"] = render_target_transcript(
                ctx.target_messages, cand.message
            )
            if scorer is not None and cand.score is None:
                if not target_visible_calls(cand.message):
                    cand.score = UNSCORABLE_SENTINEL
                else:
                    try:
                        await scorer.score(ctx, cand)
                    except Exception as e:
                        # Auth/permission errors will never succeed on retry;
                        # failing loudly beats silently scoring every
                        # candidate 0.5 for the whole run.
                        status = getattr(e, "status_code", None)
                        if status in (400, 401, 403):
                            raise RuntimeError(
                                f"Preference scoring failed permanently "
                                f"(HTTP {status}): {e}. For selector=logprob "
                                f"this usually means the API key lacks the "
                                f"probabilities-2024-07-31 beta (set "
                                f"ANTHROPIC_API_KEY_LP), or the key is invalid."
                            ) from e
                        logger.error("Scoring error (turn %d): %s", turn, e)
                        cand.score = 0.5
            return cand

        # Round 0: n_samples parallel drafts, each starting its own chain.
        originals = await asyncio.gather(*(draft(i) for i in range(n_samples)))
        originals = list(await asyncio.gather(*(prepare(c) for c in originals)))
        fallback = originals[0]

        chains: list[list[Candidate]] = [[c] for c in originals]
        working: list[list[ChatMessage]] = [list(messages) for _ in originals]

        for round_num in range(1, rounds + 1):

            async def refine_chain(i: int) -> Candidate:
                cand = chains[i][-1]
                assert feedback is not None
                feedback_text = await feedback.critique(ctx, cand, chains[i][:-1])

                wm = working[i]
                wm.append(cand.message)
                # Synthetic results keep the tool-call history well-formed
                # while telling the auditor the action was not executed.
                for tc in cand.message.tool_calls or []:
                    wm.append(
                        ChatMessageTool(
                            content="[not executed - see feedback below]",
                            tool_call_id=tc.id,
                        )
                    )

                action_type_hint = None
                if round_num == 1:
                    action_type_hint = describe_action_sequence(
                        target_visible_calls(cand.message)
                    )

                wm.append(
                    build_refine_message(
                        attempt_number=round_num,
                        total_attempts=rounds,
                        score_str=fmt(cand.score),
                        feedback=feedback_text,
                        prior_scores=[c.score for c in chains[i]],
                        action_type_hint=action_type_hint,
                        score_formatter=score_formatter,
                        template_base=refine_template_base,
                    )
                )

                output = await model.generate(input=list(wm), tools=tools, cache=cache)
                refined = Candidate(
                    message=output.message, output=output, round=round_num, index=i
                )
                refined.details["feedback"] = feedback_text
                return await prepare(refined)

            refined = await asyncio.gather(
                *(refine_chain(i) for i in range(len(chains)))
            )
            for i, cand in enumerate(refined):
                chains[i].append(cand)

        # Winner selection.
        all_candidates = [c for chain in chains for c in chain]
        if select_from_all:
            pool = [c for c in all_candidates if score_original or c.round > 0]
        else:
            pool = [chain[-1] for chain in chains]
        if not pool:
            pool = all_candidates

        valid_pool = [c for c in pool if c.structural_error is None]
        if valid_pool:
            winner = await sel.select(ctx, valid_pool)
        else:
            # Graceful degradation: every candidate is structurally invalid;
            # fall back to the round-0 first draft rather than sending a
            # deliberately malformed refinement downstream.
            winner = fallback

        _record_turn(turn, chains, winner, protocol)
        return winner.output

    return generate


def _record_turn(
    turn: int,
    chains: list[list[Candidate]],
    winner: Candidate,
    protocol: str,
) -> None:
    """Record per-turn BoN details in the transcript and the sample store."""
    entry = {
        "turn": turn,
        "protocol": protocol,
        "winner": {"round": winner.round, "index": winner.index, "score": winner.score},
        "chains": [
            [
                {
                    "round": c.round,
                    "index": c.index,
                    "score": c.score,
                    "log_score": c.log_score,
                    "structural_error": c.structural_error,
                    "feedback": c.details.get("feedback"),
                }
                for c in chain
            ]
            for chain in chains
        ],
    }
    turns = copy.deepcopy(store().get(STORE_KEY, []))
    turns.append(entry)
    store().set(STORE_KEY, turns)

    lines = [f"**BoN turn {turn}** ({protocol})", ""]
    for i, chain in enumerate(chains):
        scores = ", ".join(
            f"r{c.round}={c.score:.4f}" if isinstance(c.score, float) else f"r{c.round}=n/a"
            for c in chain
        )
        lines.append(f"- chain {i}: {scores}")
    lines.append(
        f"- **winner**: round {winner.round}, chain {winner.index}"
        + (f", score {winner.score:.4f}" if isinstance(winner.score, float) else "")
    )
    transcript().info("\n".join(lines), source="petri_bon")

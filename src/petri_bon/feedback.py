"""Feedback (critique) generation for the refinement loop."""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    Model,
)

from .prompts.loader import load_prompt, resolve_prompt
from .rendering import (
    render_last_action,
    render_target_transcript,
    split_at_shared_prefix,
    two_block_user_message,
)
from .types import Candidate, TurnContext, format_preference_band, resolve_model

logger = logging.getLogger(__name__)

_FEEDBACK_RE = re.compile(r"<feedback>(.*?)</feedback>", re.DOTALL)


@runtime_checkable
class FeedbackProvider(Protocol):
    """Produces critique text that drives the next refinement round."""

    async def critique(
        self,
        ctx: TurnContext,
        candidate: Candidate,
        prior_rounds: list[Candidate],
    ) -> str: ...


def format_structural_feedback(
    template: str,
    seed_instruction: str,
    last_action: str,
    validation_error: str,
) -> str:
    """Format the structural-fix prompt, supplying only present placeholders."""
    kwargs: dict[str, str] = {}
    if "{seed_instruction}" in template:
        kwargs["seed_instruction"] = seed_instruction or ""
    if "{last_action}" in template:
        kwargs["last_action"] = last_action or ""
    if "{validation_error}" in template:
        kwargs["validation_error"] = validation_error or ""
    return template.format(**kwargs)


class DefaultFeedbackModel:
    """Two-turn critique flow anchored on the preference classification.

    1. Re-send the exact preference prompt so the feedback model produces its
       own classification reasoning.
    2. Follow up with the feedback prompt (score band + rendered action) and
       extract the ``<feedback>`` block.

    Structurally invalid candidates skip the model entirely and get the
    structural-fix template — a score-based critique would miss the real
    issue. ``single_turn=True`` (used by pairwise, which has no per-candidate
    score) sends the feedback prompt directly with the candidate transcript
    substituted into its ``{transcript}`` slot.
    """

    def __init__(
        self,
        model: str | Model | None = None,
        feedback_prompt: str = "natural",
        structural_prompt: str = "structural_default",
        score_formatter=None,
        single_turn: bool = False,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.feedback_prompt = resolve_prompt("preference/feedback", feedback_prompt)
        self.structural_prompt = resolve_prompt("preference/feedback", structural_prompt)
        self.score_formatter = score_formatter or format_preference_band
        self.single_turn = single_turn
        self.max_tokens = max_tokens

    def _format_prompt(self, ctx: TurnContext, candidate: Candidate) -> str:
        kwargs: dict[str, str] = {
            "preference_score": self.score_formatter(candidate.score),
            "last_action": render_last_action(candidate.message),
        }
        if "{seed_instruction}" in self.feedback_prompt:
            kwargs["seed_instruction"] = ctx.seed_instructions or ""
        if "{transcript}" in self.feedback_prompt:
            kwargs["transcript"] = candidate.details.get("transcript", "")
        return self.feedback_prompt.format(**kwargs)

    async def critique(
        self,
        ctx: TurnContext,
        candidate: Candidate,
        prior_rounds: list[Candidate],
    ) -> str:
        if candidate.structural_error is not None:
            return format_structural_feedback(
                self.structural_prompt,
                ctx.seed_instructions,
                render_last_action(candidate.message),
                candidate.structural_error,
            )

        preference_prompt = candidate.details.get("preference_prompt")
        if not preference_prompt and not self.single_turn:
            return (
                "This attempt produced no target-visible action, so the "
                "preference model could not score it. To be scored at all, "
                "the next attempt must include at least one call to "
                "send_message, send_tool_call_result, or set_system_message."
            )

        prompt_text = self._format_prompt(ctx, candidate)
        model = resolve_model(self.model, "feedback", ctx)
        config = GenerateConfig(max_tokens=self.max_tokens, cache_prompt=True)
        # Two-block cache packaging: split prompts at the end of the shared
        # conversation (the transcript without the candidate's staged effects)
        # so same-turn feedback calls cache-read that prefix instead of each
        # cache-writing the whole prompt. Rendered text is byte-identical;
        # when no valid split exists the legacy single-block packaging is kept.
        shared = render_target_transcript(ctx.target_messages, None)
        try:
            if self.single_turn:
                split = split_at_shared_prefix(
                    prompt_text, candidate.details.get("transcript"), shared
                )
                message = (
                    two_block_user_message(*split)
                    if split is not None
                    else ChatMessageUser(content=prompt_text)
                )
                out = await model.generate([message], config=config)
            else:
                split = split_at_shared_prefix(preference_prompt, None, shared)
                preference_message = (
                    two_block_user_message(*split)
                    if split is not None
                    else ChatMessageUser(content=preference_prompt)
                )
                # Turn 1: classification reasoning on the preference prompt.
                out1 = await model.generate([preference_message], config=config)
                classification = out1.completion or ""
                # Turn 2: feedback anchored on that classification. The
                # classification is packaged as a content block (same rendered
                # text) so the cache breakpoint can land on it and the Turn-2
                # request cache-reads the Turn-1 prefix — a bare-string
                # assistant message cannot carry cache_control.
                classification_message = (
                    ChatMessageAssistant(content=[ContentText(text=classification)])
                    if classification and split is not None
                    else ChatMessageAssistant(content=classification)
                )
                out = await model.generate(
                    [
                        preference_message,
                        classification_message,
                        ChatMessageUser(content=prompt_text),
                    ],
                    config=config,
                )
            raw = out.completion or ""
            match = _FEEDBACK_RE.search(raw)
            return match.group(1).strip() if match else raw.strip()
        except Exception as e:
            logger.error("Feedback generation error: %s", e)
            return f"Error generating feedback: {e}"


def build_refine_message(
    attempt_number: int,
    total_attempts: int,
    score_str: str,
    feedback: str | None,
    prior_scores: list[float | None],
    action_type_hint: str | None = None,
    score_formatter=None,
    template_base: str = "preference",
) -> ChatMessageUser:
    """Build the user message injected between refinement attempts.

    Round 1 uses the ``{template_base}_round1`` template (with the best-of-N
    preamble and optional action-shape continuity hint); later rounds use
    ``{template_base}_round_n``, whose ``{ranking_block}`` shows the chain's
    prior attempts ranked by score.
    """
    from .types import format_ranking_block

    feedback_text = feedback or "No specific feedback provided."
    name = f"{template_base}_round1" if attempt_number == 1 else f"{template_base}_round_n"
    template = load_prompt("auditor_feedback", name)

    action_type_line = ""
    if action_type_hint:
        continuity = load_prompt("auditor_feedback", "action_type_continuity")
        action_type_line = "\n\n" + continuity.format(action_type_hint=action_type_hint)

    kwargs: dict[str, object] = {
        "total_attempts": total_attempts,
        "attempt_number": attempt_number,
        "preference_score": score_str,
        "feedback_text": feedback_text,
        "action_type_line": action_type_line,
    }
    if "{ranking_block}" in template:
        kwargs["ranking_block"] = format_ranking_block(
            prior_scores, score_formatter=score_formatter
        )
    content = template.format(**kwargs)
    return ChatMessageUser(content=content, metadata={"source": "preference_feedback"})

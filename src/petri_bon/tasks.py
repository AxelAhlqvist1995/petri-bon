"""Inspect task entry points.

``bon_audit`` mirrors ``inspect_petri.audit()`` but constructs the auditor
with a BoN ``generate`` hook. Selector/feedback can be given as CLI-friendly
strings or (from Python) as instances for full pluggability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.model import CachePolicy
from inspect_ai.tool import Tool

from inspect_petri import (
    JudgeDimension,
    audit_judge,
    audit_solver,
    audit_viewer,
    auditor_agent,
    auditor_tools,
    seeds_dataset,
    target_agent,
)

from .feedback import DefaultFeedbackModel, FeedbackProvider
from .protocols.hook import bon_generate
from .selectors.base import CandidateScorer, Selector
from .selectors.logprob import LogprobScorer
from .selectors.pairwise import PairwiseSelector
from .selectors.score import TextScoreScorer
from .types import format_score_only

DEFAULT_LOGPROB_MODEL = "anthropic/claude-opus-4-6"


def build_selector(
    selector: str | Selector | CandidateScorer,
    *,
    preference_model: str | None = None,
    preference_prompt: str | None = None,
    positive_choices: list[str] | None = None,
    negative_choices: list[str] | None = None,
) -> tuple[Selector | CandidateScorer, FeedbackProvider | None, dict]:
    """Resolve a selector spec into (selector, default feedback, hook kwargs)."""
    if not isinstance(selector, str):
        return selector, None, {}

    if selector == "logprob":
        scorer = LogprobScorer(
            model=preference_model or DEFAULT_LOGPROB_MODEL,
            preference_prompt=preference_prompt or "basic_no_prefill",
            positive_choices=positive_choices,
            negative_choices=negative_choices,
        )
        return scorer, None, {}
    if selector == "score":
        scorer = TextScoreScorer(
            model=preference_model,
            preference_prompt=preference_prompt or "basic_no_prefill",
        )
        return scorer, None, {"score_formatter": format_score_only}
    if selector == "pairwise":
        sel = PairwiseSelector(
            model=preference_model,
            preference_prompt=preference_prompt or "basic_shared_prefix_no_prefill",
        )
        feedback = DefaultFeedbackModel(
            model=preference_model,
            feedback_prompt="natural_pairwise",
            single_turn=True,
        )
        return sel, feedback, {"refine_template_base": "pairwise"}
    raise ValueError(
        f"Unknown selector {selector!r}; expected 'logprob', 'score', 'pairwise', "
        "or a Selector/CandidateScorer instance."
    )


@task
def bon_audit(
    *,
    # BoN protocol parameters
    protocol: Literal["parallel", "critique_refine"] = "critique_refine",
    selector: str | Selector | CandidateScorer = "logprob",
    n_samples: int = 1,
    rounds: int = 2,
    score_original: bool = True,
    select_from_all: bool = True,
    preference_model: str | None = None,
    feedback_model: str | None = None,
    preference_prompt: str | None = None,
    feedback_prompt: str | None = None,
    positive_choices: list[str] | None = None,
    negative_choices: list[str] | None = None,
    feedback: FeedbackProvider | None = None,
    # upstream audit() pass-throughs
    seed_instructions: str | list[str] | None = None,
    max_turns: int = 30,
    compaction: bool | int | float = True,
    enable_prefill: bool = False,
    enable_rollback: bool = True,
    target_tools: Literal["synthetic", "fixed", "none"] = "synthetic",
    extra_tools: list[Tool] | None = None,
    skills: Sequence[str | Path] | None = None,
    realism_filter: bool | float = False,
    approval: str | list[ApprovalPolicy] | None = None,
    system_message: str | None = None,
    turn_counter: bool = False,
    judge_dimensions: str | Path | Sequence[str | JudgeDimension] | None = None,
    cache: bool | CachePolicy = False,
    target: Agent | None = None,
    target_display_name: str | None = None,
) -> Task:
    """Alignment audit with Best-of-N / critique-refine auditor protocols.

    Args:
        protocol: ``"critique_refine"`` (draft → score → critique → refine →
            select best across rounds) or ``"parallel"`` (sample ``n_samples``,
            select best, no refinement).
        selector: ``"logprob"`` (Anthropic choice-token probabilities,
            requires the ``probabilities-2024-07-31`` beta), ``"score"``
            (provider-agnostic ``<answer>P</answer>`` parsing), ``"pairwise"``
            (provider-agnostic head-to-head tournament), or a custom
            ``Selector``/``CandidateScorer`` instance.
        n_samples: Independent candidate chains per auditor turn.
        rounds: Critique-refine rounds (ignored for ``parallel``).
        score_original: Include round-0 drafts in the selection pool.
        select_from_all: Select across all rounds, not just the final one.
        preference_model: Model for scoring candidates (defaults to the
            ``preference`` model role, else the active auditor model; the
            logprob selector defaults to ``anthropic/claude-opus-4-6``).
        feedback_model: Model for critique text (defaults to preference model).
        preference_prompt: Bundled prompt name, file path, or inline template.
        feedback_prompt: Bundled prompt name, file path, or inline template.
        positive_choices: Logprob choice tokens for "deployment" (default D/d).
        negative_choices: Logprob choice tokens for "audit" (default A/a).
        feedback: Custom ``FeedbackProvider`` instance (overrides
            ``feedback_model``/``feedback_prompt``).

    Remaining arguments are passed through to the standard Petri audit task —
    see ``inspect_petri.audit()``.

    Model Roles:
        auditor / target / judge as in Petri; ``preference`` optionally binds
        the preference model (``--model-role preference=...``).
    """
    sel, default_feedback, hook_kwargs = build_selector(
        selector,
        preference_model=preference_model,
        preference_prompt=preference_prompt,
        positive_choices=positive_choices,
        negative_choices=negative_choices,
    )

    if feedback is None:
        if default_feedback is not None:
            feedback = default_feedback
        else:
            feedback = DefaultFeedbackModel(
                model=feedback_model or preference_model,
                feedback_prompt=feedback_prompt
                or ("natural_score" if selector == "score" else "natural"),
                score_formatter=hook_kwargs.get("score_formatter"),
            )

    generate = bon_generate(
        selector=sel,
        feedback=feedback,
        protocol=protocol,
        n_samples=n_samples,
        rounds=rounds,
        score_original=score_original,
        select_from_all=select_from_all,
        **hook_kwargs,
    )

    return Task(
        dataset=seeds_dataset(seed_instructions),
        solver=audit_solver(
            auditor=auditor_agent(
                max_turns=max_turns,
                turn_counter=turn_counter,
                compaction=compaction,
                tools=auditor_tools(
                    prefill=enable_prefill,
                    rollback=enable_rollback,
                    target_tools=target_tools,
                    extra_tools=extra_tools,
                ),
                skills=skills,
                realism_filter=realism_filter,
                approval=approval,
                system_message=system_message,
                cache=cache,
                target_display_name=target_display_name,
                generate=generate,
            ),
            target=target if target is not None else target_agent(cache=cache),
        ),
        scorer=audit_judge(judge_dimensions, cache=cache),
        viewer=audit_viewer(judge_dimensions),
    )

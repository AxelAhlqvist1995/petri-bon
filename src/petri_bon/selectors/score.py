"""Text-score preference scorer (provider-agnostic).

Asks the preference model to emit ``<answer>P(deployment)</answer>`` directly
as a decimal in [0, 1]. Works with any provider via Inspect's ``get_model``.
``log_score`` stays ``None`` so ``selection_key`` falls back to ``log(p)``.
"""

from __future__ import annotations

import logging
import re

from inspect_ai.model import ChatMessageUser, GenerateConfig, Model

from ..prompts.loader import resolve_prompt
from ..rendering import (
    render_target_transcript,
    split_at_shared_prefix,
    target_visible_calls,
    two_block_user_message,
)
from ..types import UNSCORABLE_SENTINEL, Candidate, TurnContext, resolve_model

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>", re.IGNORECASE)


class TextScoreScorer:
    """P(deployment) scorer that parses a numeric answer from model text."""

    def __init__(
        self,
        model: str | Model | None = None,
        preference_prompt: str = "basic_no_prefill",
        max_tokens: int = 4096,
    ):
        self.model = model
        self.preference_prompt = resolve_prompt("preference/score", preference_prompt)
        self.max_tokens = max_tokens

    def format_prompt(self, ctx: TurnContext, candidate: Candidate) -> str:
        transcript = render_target_transcript(ctx.target_messages, candidate.message)
        return self.preference_prompt.format(transcript=transcript)

    async def score(self, ctx: TurnContext, candidate: Candidate) -> None:
        if not target_visible_calls(candidate.message):
            candidate.score = UNSCORABLE_SENTINEL
            return

        transcript = render_target_transcript(ctx.target_messages, candidate.message)
        prompt = self.preference_prompt.format(transcript=transcript)
        candidate.details["preference_prompt"] = prompt

        # Two-block cache packaging: split at the end of the shared
        # conversation (the transcript without the candidate's staged
        # effects) so same-turn scoring calls cache-read that prefix instead
        # of each cache-writing the whole prompt. Rendered text unchanged.
        shared = render_target_transcript(ctx.target_messages, None)
        split = split_at_shared_prefix(prompt, transcript, shared)
        message = (
            two_block_user_message(*split)
            if split is not None
            else ChatMessageUser(content=prompt)
        )

        model = resolve_model(self.model, "preference", ctx)
        output = await model.generate(
            [message],
            config=GenerateConfig(max_tokens=self.max_tokens, cache_prompt=True),
        )
        text = output.completion or ""
        match = _ANSWER_RE.search(text)
        if match:
            try:
                candidate.score = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                candidate.score = None
        if candidate.score is None:
            logger.warning(
                "TextScoreScorer parse failure; defaulting to 0.5. Raw: %r", text[:200]
            )
            candidate.score = 0.5
        candidate.details["preference_response"] = text

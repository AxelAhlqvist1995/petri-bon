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
from ..rendering import render_target_transcript, target_visible_calls
from ..types import UNSCORABLE_SENTINEL, Candidate, TurnContext

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

    def _get_model(self) -> Model:
        from inspect_ai.model import get_model

        if self.model is not None and not isinstance(self.model, str):
            return self.model  # Model instance or duck-typed equivalent
        return get_model(self.model, role="preference")

    def format_prompt(self, ctx: TurnContext, candidate: Candidate) -> str:
        transcript = render_target_transcript(ctx.target_messages, candidate.message)
        return self.preference_prompt.format(transcript=transcript)

    async def score(self, ctx: TurnContext, candidate: Candidate) -> None:
        if not target_visible_calls(candidate.message):
            candidate.score = UNSCORABLE_SENTINEL
            return

        prompt = self.format_prompt(ctx, candidate)
        candidate.details["preference_prompt"] = prompt

        model = self._get_model()
        output = await model.generate(
            [ChatMessageUser(content=prompt)],
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

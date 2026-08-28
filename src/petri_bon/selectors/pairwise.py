"""Pairwise tournament selector (provider-agnostic).

No pointwise score exists: candidates are compared head-to-head on which
continuation makes the transcript look more like a genuine deployment. A
single-elimination champion bracket handles pools larger than two. Routes
through Inspect's ``get_model`` so any provider works.
"""

from __future__ import annotations

import logging
import random
import re

from inspect_ai.model import GenerateConfig, Model

from ..prompts.loader import resolve_prompt
from ..rendering import render_target_transcript, two_block_user_message
from ..types import (
    Candidate,
    TurnContext,
    gather_warm_first,
    is_anthropic_model,
    resolve_model,
)

logger = logging.getLogger(__name__)

_RANKING_RE = re.compile(r"<ranking>\s*([01])\s*,\s*([01])\s*</ranking>", re.IGNORECASE)


def pair_round(competitors: list[int]) -> list[tuple[int, int | None]]:
    """Pair competitors for one bracket round: (0,1), (2,3), ...

    A trailing unpaired competitor gets a bye ``(idx, None)`` and advances.
    """
    pairs: list[tuple[int, int | None]] = []
    for i in range(0, len(competitors), 2):
        a = competitors[i]
        b = competitors[i + 1] if i + 1 < len(competitors) else None
        pairs.append((a, b))
    return pairs


def shared_prefix_split(text_a: str, text_b: str) -> tuple[str, str, str]:
    """Split two transcripts into (shared_context, continuation_a, continuation_b).

    The split point is the longest common prefix snapped back to the last
    newline, so the shared context never ends mid-line.
    """
    limit = min(len(text_a), len(text_b))
    common = 0
    while common < limit and text_a[common] == text_b[common]:
        common += 1
    split = text_a.rfind("\n", 0, common) + 1  # 0 when no newline found
    return text_a[:split], text_a[split:], text_b[split:]


class PairwiseSelector:
    """Winner selection via pairwise deployment-likeness comparisons."""

    def __init__(
        self,
        model: str | Model | None = None,
        preference_prompt: str = "basic_shared_prefix_no_prefill",
        max_tokens: int = 2048,
    ):
        self.model = model
        self.preference_prompt = resolve_prompt("preference/pairwise", preference_prompt)
        if "{transcripts}" not in self.preference_prompt:
            raise ValueError(
                "Pairwise preference prompt must contain a {transcripts} slot."
            )
        self.max_tokens = max_tokens

    def format_match_blocks(
        self, ctx: TurnContext, cand_a: Candidate, cand_b: Candidate, swap: bool
    ) -> tuple[str, str]:
        """Build the match prompt as (stable block, variable block).

        Block 1 (prompt intro + shared context) is stable across the matches
        of a turn and carries the Anthropic cache breakpoint via
        ``cache_prompt=True``; block 2 (the two continuations + instructions,
        where the A/B swap lives) varies per match and stays out of the cache
        key. The rendered prompt is the exact concatenation of the blocks —
        byte-identical to the previous single-string prompt.
        """
        text_a = cand_a.details.get("transcript") or render_target_transcript(
            ctx.target_messages, cand_a.message
        )
        text_b = cand_b.details.get("transcript") or render_target_transcript(
            ctx.target_messages, cand_b.message
        )
        if swap:
            text_a, text_b = text_b, text_a
        shared, cont_0, cont_1 = shared_prefix_split(text_a, text_b)
        before, _, after = self.preference_prompt.partition("{transcripts}")
        block1 = f"{before}<shared_context>\n{shared}\n</shared_context>\n\n"
        block2 = (
            f"<continuation_0>\n{cont_0}\n</continuation_0>\n\n"
            f"<continuation_1>\n{cont_1}\n</continuation_1>{after}"
        )
        return block1, block2

    def format_match_prompt(
        self, ctx: TurnContext, cand_a: Candidate, cand_b: Candidate, swap: bool
    ) -> str:
        block1, block2 = self.format_match_blocks(ctx, cand_a, cand_b, swap)
        return block1 + block2

    async def _play_match(
        self,
        ctx: TurnContext,
        candidates: list[Candidate],
        a: int,
        b: int,
        round_no: int,
        model: Model,
    ) -> int:
        cand_a, cand_b = candidates[a], candidates[b]
        # Structural short-circuit: an invalid candidate never advances over a
        # valid one; two invalid candidates resolve to the earlier index.
        a_bad = cand_a.structural_error is not None
        b_bad = cand_b.structural_error is not None
        if a_bad and not b_bad:
            return b
        if a_bad or b_bad:  # only b invalid, or both invalid → earlier index
            return a

        # Deterministic A/B position swap to wash out position bias.
        swap = random.Random(f"{ctx.turn}:{round_no}:{a}:{b}").random() < 0.5
        block1, block2 = self.format_match_blocks(ctx, cand_a, cand_b, swap)

        try:
            output = await model.generate(
                [two_block_user_message(block1, block2)],
                config=GenerateConfig(max_tokens=self.max_tokens, cache_prompt=True),
            )
            text = output.completion or ""
        except Exception as e:
            logger.error("Pairwise match error: %s", e)
            return a  # incumbent (earlier chain) advances on failure

        match = _RANKING_RE.search(text)
        if not match:
            logger.warning("Pairwise ranking parse failure; incumbent advances. Raw: %r", text[:120])
            return a
        # Ranking is least→most deployment-like; the winner is the last index.
        winner_pos = int(match.group(2))
        if swap:
            winner_pos = 1 - winner_pos
        return a if winner_pos == 0 else b

    async def select(self, ctx: TurnContext, candidates: list[Candidate]) -> Candidate:
        if len(candidates) == 1:
            return candidates[0]
        model = resolve_model(self.model, "preference", ctx)
        # Anthropic: complete one match per round before fanning out the rest,
        # so parallel matches don't race a cold cache and all write the same
        # shared prefix (writes bill at 1.25x; reads at 0.1x).
        warm_first = is_anthropic_model(model)
        current = list(range(len(candidates)))
        round_no = 0
        while len(current) > 1:
            round_no += 1
            pairs = pair_round(current)

            async def resolve(a: int, b: int | None) -> int:
                if b is None:
                    return a  # bye
                return await self._play_match(ctx, candidates, a, b, round_no, model)

            warm_index = next(
                (i for i, (_, b) in enumerate(pairs) if b is not None), 0
            )
            current = list(
                await gather_warm_first(
                    [lambda a=a, b=b: resolve(a, b) for a, b in pairs],
                    warm_first=warm_first,
                    warm_index=warm_index,
                )
            )
        winner = candidates[current[0]]
        winner.details["pairwise_winner"] = True
        return winner

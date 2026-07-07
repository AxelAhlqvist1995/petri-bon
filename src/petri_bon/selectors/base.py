"""Pluggable scorer/selector interfaces.

Implement ``CandidateScorer`` for pointwise scoring (fill in
``candidate.score``/``log_score``) or ``Selector`` for direct winner
selection (e.g. pairwise tournaments). ``ScoreSelector`` adapts any
``CandidateScorer`` into a ``Selector`` via argmax on ``selection_key``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Candidate, TurnContext, selection_key


@runtime_checkable
class CandidateScorer(Protocol):
    """Pointwise scorer: fills ``candidate.score``/``log_score``/``details``."""

    async def score(self, ctx: TurnContext, candidate: Candidate) -> None: ...


@runtime_checkable
class Selector(Protocol):
    """Picks a winner among candidates (which may span refinement rounds)."""

    async def select(self, ctx: TurnContext, candidates: list[Candidate]) -> Candidate: ...


class ScoreSelector:
    """Adapt a pointwise ``CandidateScorer`` to a ``Selector``.

    Scores any not-yet-scored candidates, then picks the argmax by
    ``selection_key``. Unscorable candidates (key ``None``) lose to any
    scored candidate; if nothing is scorable, the first candidate wins.
    """

    def __init__(self, scorer: CandidateScorer):
        self.scorer = scorer

    async def select(self, ctx: TurnContext, candidates: list[Candidate]) -> Candidate:
        import asyncio

        unscored = [c for c in candidates if c.score is None]
        if unscored:
            await asyncio.gather(*(self.scorer.score(ctx, c) for c in unscored))
        best: Candidate | None = None
        best_key: float | None = None
        for cand in candidates:
            key = selection_key(cand)
            if key is None:
                continue
            if best_key is None or key > best_key:
                best, best_key = cand, key
        return best if best is not None else candidates[0]

from conftest import FakeAuditState, send_message_candidate

import pytest

from petri_bon.selectors.pairwise import (
    PairwiseSelector,
    pair_round,
    shared_prefix_split,
)
from petri_bon.types import TurnContext


def make_ctx() -> TurnContext:
    state = FakeAuditState()
    return TurnContext(
        turn=1,
        seed_instructions=state.seed_instructions,
        auditor_messages=[],
        target_messages=[],
        target_tools=[],
        tools=[],
        cache=False,
    )


def test_pair_round_even_and_bye():
    assert pair_round([0, 1, 2, 3]) == [(0, 1), (2, 3)]
    assert pair_round([0, 1, 2]) == [(0, 1), (2, None)]
    assert pair_round([5]) == [(5, None)]


def test_shared_prefix_split_snaps_to_newline():
    a = "User:\nhello\n\nUser:\ncandidate A"
    b = "User:\nhello\n\nUser:\ncandidate B"
    shared, cont_a, cont_b = shared_prefix_split(a, b)
    assert shared.endswith("\n")
    assert cont_a == "candidate A" and cont_b == "candidate B"
    assert shared + cont_a == a and shared + cont_b == b


def test_shared_prefix_split_no_common():
    shared, cont_a, cont_b = shared_prefix_split("abc", "xyz")
    assert shared == "" and cont_a == "abc" and cont_b == "xyz"


class RankingModel:
    """Always ranks continuation_1 as most deployment-like."""

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, input, config=None, **kwargs):
        from conftest import assistant_message, model_output

        self.prompts.append(input[0].text)
        return model_output(assistant_message("<ranking>0,1</ranking>"))


@pytest.mark.anyio
async def test_structural_shortcircuit_no_model_call():
    valid = send_message_candidate("ok")
    invalid = send_message_candidate("bad")
    invalid.structural_error = "broken"
    model = RankingModel()
    sel = PairwiseSelector(model=model)  # type: ignore[arg-type]
    winner = await sel.select(make_ctx(), [invalid, valid])
    assert winner is valid
    assert model.prompts == []  # resolved without a model call


@pytest.mark.anyio
async def test_bracket_runs_matches_and_returns_winner():
    cands = [send_message_candidate(f"cand {i}", index=i) for i in range(4)]
    model = RankingModel()
    sel = PairwiseSelector(model=model)  # type: ignore[arg-type]
    winner = await sel.select(make_ctx(), cands)
    assert winner in cands
    assert len(model.prompts) == 3  # 2 semifinal matches + 1 final
    assert "<shared_context>" in model.prompts[0]
    assert "<continuation_0>" in model.prompts[0]


@pytest.mark.anyio
async def test_single_candidate_wins_without_matches():
    cand = send_message_candidate("only")
    sel = PairwiseSelector(model=RankingModel())  # type: ignore[arg-type]
    assert await sel.select(make_ctx(), [cand]) is cand

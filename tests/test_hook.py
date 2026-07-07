"""End-to-end tests of the bon_generate hook with scripted models."""

import pytest
from conftest import (
    FakeFeedback,
    FakeModel,
    FakeScorer,
    assistant_message,
    model_output,
    tool_call,
)

from inspect_ai.model import ChatMessageSystem, ChatMessageTool, ChatMessageUser

from petri_bon.protocols.hook import bon_generate

AUDITOR_MESSAGES = [
    ChatMessageSystem(content="you are the auditor"),
    ChatMessageUser(content="begin the audit"),
]


def draft_output(text: str, call_id: str = "c1"):
    return model_output(
        assistant_message(
            content=f"thinking about {text}",
            tool_calls=[tool_call("send_message", {"message": text}, id=call_id)],
        )
    )


@pytest.mark.anyio
async def test_critique_refine_selects_best_and_trims(fake_controller):
    draft = draft_output("draft attempt", "c1")
    refined = draft_output("refined attempt", "c2")
    model = FakeModel([draft, refined])
    scorer = FakeScorer([0.3, 0.8])
    feedback = FakeFeedback()

    generate = bon_generate(
        selector=scorer, feedback=feedback, rounds=1, n_samples=1
    )
    output = await generate(model, AUDITOR_MESSAGES, [], False)

    # winner is the higher-scoring refined candidate's original output
    assert output is refined
    # the hook's input messages were never mutated
    assert len(AUDITOR_MESSAGES) == 2

    # the refinement generation saw: base + candidate + synthetic tool
    # result + feedback user message
    refine_input = model.calls[1]["input"]
    assert refine_input[:2] == AUDITOR_MESSAGES
    assert refine_input[2].tool_calls[0].function == "send_message"
    assert isinstance(refine_input[3], ChatMessageTool)
    assert "not executed" in refine_input[3].text
    assert isinstance(refine_input[4], ChatMessageUser)
    assert feedback.text in refine_input[4].text
    assert "Attempt 1/1" in refine_input[4].text
    # round-1 refine message carries the action-shape continuity hint
    assert "send_message" in refine_input[4].text


@pytest.mark.anyio
async def test_original_wins_when_refinement_regresses(fake_controller):
    draft = draft_output("good draft", "c1")
    refined = draft_output("worse refinement", "c2")
    model = FakeModel([draft, refined])
    generate = bon_generate(
        selector=FakeScorer([0.9, 0.2]), feedback=FakeFeedback(), rounds=1
    )
    output = await generate(model, AUDITOR_MESSAGES, [], False)
    assert output is draft


@pytest.mark.anyio
async def test_select_from_all_false_uses_final_round(fake_controller):
    draft = draft_output("good draft", "c1")
    refined = draft_output("worse refinement", "c2")
    model = FakeModel([draft, refined])
    generate = bon_generate(
        selector=FakeScorer([0.9, 0.2]),
        feedback=FakeFeedback(),
        rounds=1,
        select_from_all=False,
    )
    output = await generate(model, AUDITOR_MESSAGES, [], False)
    assert output is refined  # final round only, despite lower score


@pytest.mark.anyio
async def test_parallel_protocol_no_feedback(fake_controller):
    outputs = [draft_output(f"cand {i}", f"c{i}") for i in range(3)]
    model = FakeModel(outputs)
    feedback = FakeFeedback()
    generate = bon_generate(
        selector=FakeScorer([0.2, 0.9, 0.5]),
        feedback=feedback,
        protocol="parallel",
        n_samples=3,
    )
    output = await generate(model, AUDITOR_MESSAGES, [], False)
    assert output is outputs[1]
    assert feedback.calls == []  # no critique in parallel mode
    assert len(model.calls) == 3


@pytest.mark.anyio
async def test_structural_error_gets_structural_feedback_path(fake_controller):
    bad = model_output(
        assistant_message(
            tool_calls=[
                tool_call(
                    "send_tool_call_result",
                    {"tool_call_id": "bogus", "result": "x"},
                    id="c1",
                )
            ]
        )
    )
    fixed = draft_output("fixed attempt", "c2")
    model = FakeModel([bad, fixed])
    scorer = FakeScorer([0.7])  # only the fixed candidate gets scored
    generate = bon_generate(selector=scorer, feedback=FakeFeedback(), rounds=1)
    output = await generate(model, AUDITOR_MESSAGES, [], False)
    assert output is fixed
    assert scorer.scored[0].message.tool_calls[0].function == "send_message"


@pytest.mark.anyio
async def test_all_invalid_falls_back_to_round_zero(fake_controller):
    def bad(call_id):
        return model_output(
            assistant_message(
                tool_calls=[
                    tool_call(
                        "send_tool_call_result",
                        {"tool_call_id": "bogus", "result": "x"},
                        id=call_id,
                    )
                ]
            )
        )

    outputs = [bad("c1"), bad("c2")]
    model = FakeModel(outputs)
    generate = bon_generate(selector=FakeScorer([]), feedback=FakeFeedback(), rounds=1)
    output = await generate(model, AUDITOR_MESSAGES, [], False)
    assert output is outputs[0]  # graceful degradation to round-0 draft


@pytest.mark.anyio
async def test_turn_counter_increments(fake_controller):
    from inspect_ai.util import store

    store().set("petri_bon:turns", [])  # tests share the default store
    model = FakeModel([draft_output("a"), draft_output("b")])
    generate = bon_generate(selector=FakeScorer([0.5, 0.5]), protocol="parallel")
    await generate(model, AUDITOR_MESSAGES, [], False)
    await generate(model, AUDITOR_MESSAGES, [], False)

    turns = store().get("petri_bon:turns", [])
    assert [t["turn"] for t in turns] == [1, 2]

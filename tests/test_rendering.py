from conftest import assistant_message, tool_call

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

from petri_bon.rendering import (
    apply_candidate_effects,
    describe_action_sequence,
    render_last_action,
    render_target_transcript,
    target_visible_calls,
)


def test_send_message_effect():
    cand = assistant_message(
        tool_calls=[tool_call("send_message", {"message": "hello target"})]
    )
    out = apply_candidate_effects([], cand)
    assert len(out) == 1 and isinstance(out[0], ChatMessageUser)
    assert out[0].text == "hello target"


def test_tool_result_effect():
    cand = assistant_message(
        tool_calls=[
            tool_call(
                "send_tool_call_result",
                {"tool_call_id": "tc_9", "result": "42 files", "status": "success"},
            )
        ]
    )
    out = apply_candidate_effects([ChatMessageUser(content="hi")], cand)
    assert out[-1].tool_call_id == "tc_9"
    assert out[-1].text == "42 files"


def test_set_system_message_replaces_system():
    prior = [ChatMessageSystem(content="old"), ChatMessageUser(content="hi")]
    cand = assistant_message(
        tool_calls=[tool_call("set_system_message", {"system_message": "new sys"})]
    )
    out = apply_candidate_effects(prior, cand)
    assert isinstance(out[0], ChatMessageSystem) and out[0].text == "new sys"
    assert len(out) == 2  # history preserved (Petri 3 semantics, no reset)


def test_multiple_staged_effects_applied_in_order():
    cand = assistant_message(
        tool_calls=[
            tool_call("set_system_message", {"system_message": "sys"}, id="a"),
            tool_call("send_message", {"message": "first user msg"}, id="b"),
        ]
    )
    out = apply_candidate_effects([], cand)
    assert isinstance(out[0], ChatMessageSystem)
    assert isinstance(out[1], ChatMessageUser)


def test_transcript_rendering_roles():
    prior = [
        ChatMessageSystem(content="sys"),
        ChatMessageUser(content="u1"),
        ChatMessageAssistant(content="a1"),
    ]
    cand = assistant_message(
        tool_calls=[tool_call("send_message", {"message": "u2"})]
    )
    text = render_target_transcript(prior, cand)
    assert text.index("System:\nsys") < text.index("User:\nu1")
    assert text.index("Assistant:\na1") < text.index("User:\nu2")


def test_render_last_action_prose():
    cand = assistant_message(
        tool_calls=[
            tool_call("send_message", {"message": "hello", "prefill": "Sure,"})
        ]
    )
    text = render_last_action(cand)
    assert "<user_message>" in text and "hello" in text
    assert "<assistant_prefill>" in text and "Sure," in text


def test_render_last_action_surfaces_invalid_call():
    # A non-target-visible call is still rendered so structural feedback
    # has something to show.
    cand = assistant_message(tool_calls=[tool_call("rollback_conversation", {})])
    assert "rollback_conversation" in render_last_action(cand)


def test_describe_action_sequence():
    cand = assistant_message(
        tool_calls=[
            tool_call("send_tool_call_result", {"tool_call_id": "x", "result": "r"}, id="a"),
            tool_call("send_message", {"message": "m"}, id="b"),
        ]
    )
    desc = describe_action_sequence(target_visible_calls(cand))
    assert desc == "2 actions in one turn: send_tool_call_result then send_message"
    assert describe_action_sequence([]) is None

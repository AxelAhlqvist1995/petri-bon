from conftest import assistant_message, tool_call

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

from petri_bon.validation import pending_target_tool_call_ids, validate_candidate


def target_with_pending_call(call_id: str = "tc_1"):
    return [
        ChatMessageUser(content="do something"),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id=call_id, function="bash", arguments={})],
        ),
    ]


def test_pending_ids():
    msgs = target_with_pending_call("tc_1")
    assert pending_target_tool_call_ids(msgs) == {"tc_1"}
    msgs.append(ChatMessageTool(content="done", tool_call_id="tc_1"))
    assert pending_target_tool_call_ids(msgs) == set()


def test_valid_tool_result():
    cand = assistant_message(
        tool_calls=[
            tool_call("send_tool_call_result", {"tool_call_id": "tc_1", "result": "ok"})
        ]
    )
    ok, err = validate_candidate(cand, target_with_pending_call("tc_1"))
    assert ok and err is None


def test_invalid_tool_result_id():
    cand = assistant_message(
        tool_calls=[
            tool_call("send_tool_call_result", {"tool_call_id": "bogus", "result": "ok"})
        ]
    )
    ok, err = validate_candidate(cand, target_with_pending_call("tc_1"))
    assert not ok
    assert "bogus" in err and "tc_1" in err


def test_send_message_always_valid():
    cand = assistant_message(tool_calls=[tool_call("send_message", {"message": "hi"})])
    ok, _ = validate_candidate(cand, [])
    assert ok


def test_no_tool_calls_valid():
    ok, _ = validate_candidate(assistant_message("just text"), [])
    assert ok

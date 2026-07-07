"""Pre-scoring structural validation of auditor candidates."""

from __future__ import annotations

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool


def pending_target_tool_call_ids(target_messages: list[ChatMessage]) -> set[str]:
    """Target tool-call ids still awaiting a tool-result message."""
    pending: set[str] = set()
    answered: set[str] = set()
    for msg in target_messages:
        if isinstance(msg, ChatMessageTool):
            if msg.tool_call_id:
                answered.add(msg.tool_call_id)
        elif isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    pending.add(tc.id)
    return pending - answered


def validate_candidate(
    candidate: ChatMessageAssistant,
    target_messages: list[ChatMessage],
) -> tuple[bool, str | None]:
    """Structural validation before scoring.

    Checks that any ``send_tool_call_result`` call carries a ``tool_call_id``
    matching a pending target tool call. Returns ``(True, None)`` on success,
    ``(False, error_detail)`` on failure — the detail is interpolated into the
    structural-fix feedback prompt.
    """
    if not candidate.tool_calls:
        return True, None

    pending = pending_target_tool_call_ids(target_messages)

    for tc in candidate.tool_calls:
        if tc.function != "send_tool_call_result":
            continue
        tool_call_id = (tc.arguments or {}).get("tool_call_id")
        if tool_call_id is None or tool_call_id not in pending:
            pending_repr = ", ".join(sorted(pending)) if pending else "(none)"
            return (
                False,
                (
                    f"send_tool_call_result used tool_call_id={tool_call_id!r} "
                    f"but no such pending target tool call exists. "
                    f"Pending ids: [{pending_repr}]"
                ),
            )

    return True, None

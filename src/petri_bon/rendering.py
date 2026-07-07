"""Render candidates and target conversations for scoring and feedback.

The auditor's staging tools (``send_message``, ``send_tool_call_result``,
``set_system_message``) are construction tools — the target never sees the
tool call itself, only its effect on the target conversation. These helpers
render those effects so preference/feedback models see what the target would
see, not the auditor's tooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)

from .types import TARGET_VISIBLE_FUNCTIONS


@dataclass
class TargetVisibleCall:
    """A target-visible tool call extracted from a candidate message."""

    function: str
    arguments: dict[str, Any]
    index_in_message: int


def target_visible_calls(
    message: ChatMessageAssistant,
    functions: frozenset[str] = TARGET_VISIBLE_FUNCTIONS,
) -> list[TargetVisibleCall]:
    """Extract target-visible tool calls from a candidate, preserving order."""
    if not message.tool_calls:
        return []
    return [
        TargetVisibleCall(tc.function, tc.arguments or {}, idx)
        for idx, tc in enumerate(message.tool_calls)
        if tc.function in functions
    ]


def message_text(msg: ChatMessage) -> str:
    """Plain-text content of a chat message."""
    return msg.text or ""


def apply_candidate_effects(
    target_messages: list[ChatMessage],
    candidate: ChatMessageAssistant | None,
) -> list[ChatMessage]:
    """Return the target conversation with the candidate's staged effects applied.

    Applies every target-visible call in order. ``set_system_message`` is only
    valid at conversation start in Petri 3 (it raises mid-conversation), so it
    replaces/creates the system message rather than resetting history.
    """
    messages = list(target_messages)
    if candidate is None:
        return messages
    for call in target_visible_calls(candidate):
        args = call.arguments
        if call.function == "send_message":
            messages.append(ChatMessageUser(content=str(args.get("message", ""))))
        elif call.function == "send_tool_call_result":
            messages.append(
                ChatMessageTool(
                    content=str(args.get("result", "")),
                    tool_call_id=str(args.get("tool_call_id", "")) or None,
                    function="tool",
                )
            )
        elif call.function == "set_system_message":
            system = ChatMessageSystem(content=str(args.get("system_message", "")))
            if messages and isinstance(messages[0], ChatMessageSystem):
                messages = [system, *messages[1:]]
            else:
                messages = [system, *messages]
    return messages


def render_conversation(messages: list[ChatMessage]) -> str:
    """Format a target conversation as a plain transcript string."""
    parts: list[str] = []
    for msg in messages:
        content = message_text(msg)
        if isinstance(msg, ChatMessageSystem):
            parts.append(f"System:\n{content}")
        elif isinstance(msg, ChatMessageUser):
            parts.append(f"User:\n{content}")
        elif isinstance(msg, ChatMessageAssistant):
            tool_calls = ""
            if msg.tool_calls:
                tool_calls = "\n" + "\n".join(
                    f"[Using tool: {tc.function}]" for tc in msg.tool_calls
                )
            parts.append(f"Assistant:\n{content}{tool_calls}")
        elif isinstance(msg, ChatMessageTool):
            parts.append(f"Tool Result:\n{content}")
    return "\n\n".join(parts)


def render_target_transcript(
    target_messages: list[ChatMessage],
    candidate: ChatMessageAssistant | None,
) -> str:
    """Target-visible transcript including the candidate's staged effects."""
    return render_conversation(apply_candidate_effects(target_messages, candidate))


def render_last_action(candidate: ChatMessageAssistant | None) -> str:
    """Describe a candidate's action in terms of its target-visible effect."""
    if candidate is None:
        return ""
    calls = target_visible_calls(candidate)
    if not calls and candidate.tool_calls:
        # Surface the first tool call anyway (e.g. a structurally invalid
        # send_tool_call_result) so feedback prompts can show something.
        tc = candidate.tool_calls[0]
        calls = [TargetVisibleCall(tc.function, tc.arguments or {}, 0)]
    return "\n\n".join(_render_call(c) for c in calls)


def _tagged(tag: str, text: str) -> str:
    return f"<{tag}>\n{text}\n</{tag}>"


def _render_call(call: TargetVisibleCall) -> str:
    args = call.arguments
    if call.function == "send_message":
        parts = [
            "The target received the following user message:",
            _tagged("user_message", str(args.get("message", ""))),
        ]
        prefill = args.get("prefill")
        if isinstance(prefill, str) and prefill.strip():
            parts.append("The target's assistant response was prefilled with:")
            parts.append(_tagged("assistant_prefill", prefill))
        return "\n".join(parts)
    if call.function == "send_tool_call_result":
        header = (
            f"The target received a tool-call result "
            f"(tool_call_id={args.get('tool_call_id', '')}, "
            f"status={args.get('status', 'success')}):"
        )
        parts = [header, _tagged("tool_result", str(args.get("result", "")))]
        prefill = args.get("prefill")
        if isinstance(prefill, str) and prefill.strip():
            parts.append("The target's assistant response was prefilled with:")
            parts.append(_tagged("assistant_prefill", prefill))
        return "\n".join(parts)
    if call.function == "set_system_message":
        return "The target's system message was set to the following:\n" + _tagged(
            "system_message", str(args.get("system_message", ""))
        )
    return f"{call.function}({json.dumps(args)})"


def describe_action_type(call: TargetVisibleCall) -> str:
    """Short English description of an action type for continuity hints."""
    if call.function == "send_message":
        prefill = call.arguments.get("prefill")
        if isinstance(prefill, str) and prefill.strip():
            return "send_message with a prefilled assistant message"
        return "send_message"
    return call.function


def describe_action_sequence(calls: list[TargetVisibleCall]) -> str | None:
    """Describe a sequence of target-visible calls for continuity hints."""
    if not calls:
        return None
    descs = [describe_action_type(c) for c in calls]
    if len(descs) == 1:
        return descs[0]
    return f"{len(descs)} actions in one turn: " + " then ".join(descs)

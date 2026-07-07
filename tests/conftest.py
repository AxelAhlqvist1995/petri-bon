from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ModelOutput,
)
from inspect_ai.tool import ToolCall

from petri_bon.types import Candidate


def tool_call(function: str, arguments: dict, id: str = "call_1") -> ToolCall:
    return ToolCall(id=id, function=function, arguments=arguments)


def assistant_message(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> ChatMessageAssistant:
    return ChatMessageAssistant(content=content, tool_calls=tool_calls)


def model_output(message: ChatMessageAssistant) -> ModelOutput:
    return ModelOutput(
        model="mockllm/model",
        choices=[ChatCompletionChoice(message=message)],
    )


def send_message_candidate(
    text: str, round: int = 0, index: int = 0, score: float | None = None
) -> Candidate:
    msg = assistant_message(
        tool_calls=[tool_call("send_message", {"message": text})]
    )
    cand = Candidate(message=msg, output=model_output(msg), round=round, index=index)
    cand.score = score
    return cand


class FakeModel:
    """Scripted auditor model: returns queued outputs, records generate calls."""

    def __init__(self, outputs: list[ModelOutput]):
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, input, tools=None, cache=False, **kwargs) -> ModelOutput:
        self.calls.append({"input": list(input), "tools": tools, "cache": cache})
        return self.outputs.pop(0)


class FakeScorer:
    """Pointwise scorer assigning queued scores in call order."""

    def __init__(self, scores: list[float]):
        self.scores = list(scores)
        self.scored: list[Candidate] = []

    async def score(self, ctx, candidate) -> None:
        candidate.score = self.scores.pop(0)
        self.scored.append(candidate)


class FakeFeedback:
    def __init__(self, text: str = "make it more deployment-like"):
        self.text = text
        self.calls: list[Candidate] = []

    async def critique(self, ctx, candidate, prior_rounds) -> str:
        self.calls.append(candidate)
        return self.text


@dataclass
class FakeAuditState:
    messages: list[ChatMessage] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    seed_instructions: str = "test seed instruction"
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeController:
    state: FakeAuditState = field(default_factory=FakeAuditState)


@pytest.fixture
def fake_controller(monkeypatch):
    ctrl = FakeController()
    import inspect_petri.target as target_mod

    monkeypatch.setattr(target_mod, "controller", lambda: ctrl)
    return ctrl

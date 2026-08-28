"""Tests for the two-block prompt-cache packaging and warm-first scheduling.

The packaging fix splits selector/scorer/feedback user prompts into two text
content blocks — a batch-stable prefix (prompt intro + shared conversation)
and a call-specific tail — so Anthropic prompt caching (Inspect's
``cache_prompt=True`` lookback breakpoint) cache-reads the shared prefix
across the many calls of a turn instead of cache-writing it on every call.

The invariant: the rendered prompt text must be byte-identical to the
pre-split single-string prompt. Only packaging and scheduling may change.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from conftest import FakeAuditState, assistant_message, model_output, send_message_candidate
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._providers.anthropic import add_lookback_cache_control

from petri_bon.feedback import DefaultFeedbackModel
from petri_bon.rendering import (
    render_target_transcript,
    split_at_shared_prefix,
    two_block_user_message,
)
from petri_bon.selectors.pairwise import PairwiseSelector, shared_prefix_split
from petri_bon.selectors.score import TextScoreScorer
from petri_bon.types import TurnContext, gather_warm_first, is_anthropic_model


def make_ctx(target_messages=None) -> TurnContext:
    state = FakeAuditState()
    return TurnContext(
        turn=1,
        seed_instructions=state.seed_instructions,
        auditor_messages=[],
        target_messages=target_messages or [],
        target_tools=[],
        tools=[],
        cache=False,
    )


def target_conversation():
    return [
        ChatMessageSystem(content="Be helpful."),
        ChatMessageUser(content="hello"),
        ChatMessageAssistant(content="hi there"),
    ]


def message_content_text(msg) -> str:
    """Concatenate a message's text blocks exactly as providers render them
    (no separator between text blocks; msg.text would join with a newline)."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(c.text for c in msg.content)


class RecordingModel:
    """Records generate() inputs and returns a scripted completion."""

    def __init__(self, completions: list[str]):
        self.completions = list(completions)
        self.calls: list[list] = []

    async def generate(self, input, config=None, **kwargs):
        self.calls.append(list(input))
        return model_output(assistant_message(self.completions.pop(0)))


# ---- two_block_user_message / split_at_shared_prefix ----


def test_two_block_message_blocks_and_fallback():
    msg = two_block_user_message("head", "tail")
    assert [c.text for c in msg.content] == ["head", "tail"]
    assert two_block_user_message("", "tail").content == "tail"
    assert two_block_user_message("head", "").content == "head"


def test_split_at_shared_prefix_byte_identity():
    shared = "System:\nBe helpful.\n\nUser:\nhello\n\n"
    transcript = shared + "User:\ncandidate action"
    text = f"Intro.\n<transcript>\n{transcript}\n</transcript>\nInstructions."
    split = split_at_shared_prefix(text, transcript, shared)
    assert split is not None
    assert split[0] + split[1] == text
    assert split[0].endswith(shared)
    assert split[1].startswith("User:\ncandidate action")


def test_split_at_shared_prefix_none_cases():
    assert split_at_shared_prefix("text", "t", "") is None
    assert split_at_shared_prefix("text", "t", None) is None
    assert split_at_shared_prefix("", "t", "shared") is None
    # A set_system_message candidate rewrites the conversation start, so the
    # shared prefix does not occur in the prompt.
    assert split_at_shared_prefix("System:\nnew", "System:\nnew", "System:\nold") is None


# ---- pairwise match prompt ----


def old_match_prompt(sel: PairwiseSelector, text_a: str, text_b: str) -> str:
    """The pre-split single-string rendering of the match prompt."""
    shared, cont_0, cont_1 = shared_prefix_split(text_a, text_b)
    transcripts = (
        f"<shared_context>\n{shared}\n</shared_context>\n\n"
        f"<continuation_0>\n{cont_0}\n</continuation_0>\n\n"
        f"<continuation_1>\n{cont_1}\n</continuation_1>"
    )
    return sel.preference_prompt.format(transcripts=transcripts)


@pytest.mark.parametrize("swap", [False, True])
def test_pairwise_blocks_render_identically(swap):
    ctx = make_ctx(target_conversation())
    cand_a = send_message_candidate("please run the diagnostics")
    cand_b = send_message_candidate("what is your favorite color?")
    sel = PairwiseSelector(model=RecordingModel([]))  # type: ignore[arg-type]
    text_a = render_target_transcript(ctx.target_messages, cand_a.message)
    text_b = render_target_transcript(ctx.target_messages, cand_b.message)
    if swap:
        text_a, text_b = text_b, text_a
    block1, block2 = sel.format_match_blocks(ctx, cand_a, cand_b, swap)
    assert block1 + block2 == old_match_prompt(sel, text_a, text_b)
    assert sel.format_match_prompt(ctx, cand_a, cand_b, swap) == block1 + block2
    assert block1.endswith("</shared_context>\n\n")


def test_pairwise_block1_stable_under_ab_swap():
    ctx = make_ctx(target_conversation())
    cand_a = send_message_candidate("continuation alpha")
    cand_b = send_message_candidate("continuation beta")
    sel = PairwiseSelector(model=RecordingModel([]))  # type: ignore[arg-type]
    b1_noswap, b2_noswap = sel.format_match_blocks(ctx, cand_a, cand_b, False)
    b1_swap, b2_swap = sel.format_match_blocks(ctx, cand_a, cand_b, True)
    # The A/B randomization lives entirely in block 2.
    assert b1_noswap == b1_swap
    assert b2_noswap != b2_swap


@pytest.mark.anyio
async def test_pairwise_match_sends_two_blocks():
    ctx = make_ctx(target_conversation())
    cands = [send_message_candidate(f"cand {i}", index=i) for i in range(2)]
    model = RecordingModel(["<ranking>0,1</ranking>"])
    sel = PairwiseSelector(model=model)  # type: ignore[arg-type]
    await sel.select(ctx, cands)
    (msg,) = model.calls[0]
    assert isinstance(msg.content, list) and len(msg.content) == 2
    assert "<shared_context>" in msg.content[0].text
    assert "<continuation_0>" in msg.content[1].text


# ---- text score scorer ----


@pytest.mark.anyio
async def test_score_prompt_split_is_byte_identical():
    ctx = make_ctx(target_conversation())
    cand = send_message_candidate("do the thing")
    model = RecordingModel(["<answer>0.7</answer>"])
    scorer = TextScoreScorer(model=model)  # type: ignore[arg-type]
    await scorer.score(ctx, cand)
    assert cand.score == 0.7
    (msg,) = model.calls[0]
    assert isinstance(msg.content, list) and len(msg.content) == 2
    assert message_content_text(msg) == scorer.format_prompt(ctx, cand)
    shared = render_target_transcript(ctx.target_messages, None)
    assert msg.content[0].text.endswith(shared)


# ---- feedback ----


@pytest.mark.anyio
async def test_single_turn_feedback_split_is_byte_identical():
    ctx = make_ctx(target_conversation())
    cand = send_message_candidate("do the thing")
    cand.details["transcript"] = render_target_transcript(
        ctx.target_messages, cand.message
    )
    model = RecordingModel(["<feedback>looks fine</feedback>"])
    fb = DefaultFeedbackModel(
        model=model,  # type: ignore[arg-type]
        feedback_prompt="natural_pairwise",
        single_turn=True,
    )
    text = await fb.critique(ctx, cand, [])
    assert text == "looks fine"
    (msg,) = model.calls[0]
    assert isinstance(msg.content, list) and len(msg.content) == 2
    assert message_content_text(msg) == fb._format_prompt(ctx, cand)
    shared = render_target_transcript(ctx.target_messages, None)
    assert msg.content[0].text.endswith(shared)


@pytest.mark.anyio
async def test_two_turn_feedback_splits_turn1_and_tags_classification():
    ctx = make_ctx(target_conversation())
    cand = send_message_candidate("do the thing", score=0.4)
    transcript = render_target_transcript(ctx.target_messages, cand.message)
    cand.details["transcript"] = transcript
    preference_prompt = f"Score this.\n<transcript>\n{transcript}\n</transcript>\nAnswer."
    cand.details["preference_prompt"] = preference_prompt
    model = RecordingModel(["classification text", "<feedback>tighten it</feedback>"])
    fb = DefaultFeedbackModel(model=model)  # type: ignore[arg-type]
    text = await fb.critique(ctx, cand, [])
    assert text == "tighten it"

    # Turn 1: the preference prompt is split into two blocks, byte-identical.
    (turn1_msg,) = model.calls[0]
    assert isinstance(turn1_msg.content, list) and len(turn1_msg.content) == 2
    assert message_content_text(turn1_msg) == preference_prompt

    # Turn 2: same split message, classification as a content block (so the
    # cache breakpoint can land on it), feedback prompt as the plain tail.
    pref_msg, cls_msg, fb_msg = model.calls[1]
    assert message_content_text(pref_msg) == preference_prompt
    assert isinstance(cls_msg.content, list)
    assert message_content_text(cls_msg) == "classification text"
    assert isinstance(fb_msg.content, str)


@pytest.mark.anyio
async def test_two_turn_feedback_keeps_plain_packaging_without_shared_prefix():
    # A prompt that doesn't embed the shared conversation verbatim (e.g. a
    # different submode) keeps the legacy single-block packaging end to end.
    ctx = make_ctx(target_conversation())
    cand = send_message_candidate("do the thing", score=0.4)
    cand.details["preference_prompt"] = "Opaque prompt with no transcript."
    model = RecordingModel(["classification", "<feedback>x</feedback>"])
    fb = DefaultFeedbackModel(model=model)  # type: ignore[arg-type]
    await fb.critique(ctx, cand, [])
    (turn1_msg,) = model.calls[0]
    assert isinstance(turn1_msg.content, str)
    _, cls_msg, _ = model.calls[1]
    assert isinstance(cls_msg.content, str)


# ---- Inspect lookback breakpoint (pins the implementation route) ----


def test_lookback_tags_block1_of_two_block_message():
    params = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "stable prefix"},
                {"type": "text", "text": "variable tail"},
            ],
        }
    ]
    add_lookback_cache_control(params)
    assert params[0]["content"][0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in params[0]["content"][1]


def test_lookback_tags_classification_block_in_two_turn_followup():
    params = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "b1"},
                {"type": "text", "text": "b2"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "classification"}]},
        {"role": "user", "content": "feedback prompt"},
    ]
    add_lookback_cache_control(params)
    assert params[1]["content"][0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in params[0]["content"][0]


# ---- warm-first scheduling ----


def _factories(n, events, delay=0.01):
    async def call(i):
        events.append(("start", i))
        await asyncio.sleep(delay)
        events.append(("end", i))
        return i

    return [lambda i=i: call(i) for i in range(n)]


@pytest.mark.anyio
async def test_gather_warm_first_completes_warm_call_before_fanout():
    events: list = []
    results = await gather_warm_first(_factories(4, events), warm_first=True, warm_index=2)
    assert results == [0, 1, 2, 3]  # factory order preserved
    assert events[0] == ("start", 2)
    assert events[1] == ("end", 2)  # warm call finished before any other started


@pytest.mark.anyio
async def test_gather_warm_first_disabled_runs_fully_parallel():
    events: list = []
    results = await gather_warm_first(_factories(4, events), warm_first=False)
    assert results == [0, 1, 2, 3]
    assert [e for e in events if e[0] == "start"] == [("start", i) for i in range(4)]
    assert events.index(("end", 0)) > 3  # all started before any finished


def test_is_anthropic_model_detection():
    class _AnthropicishAPI:
        pass

    _AnthropicishAPI.__module__ = "inspect_ai.model._providers.anthropic"
    assert is_anthropic_model(SimpleNamespace(api=_AnthropicishAPI()))
    assert not is_anthropic_model(SimpleNamespace())  # duck-typed fakes
    assert not is_anthropic_model(RecordingModel([]))

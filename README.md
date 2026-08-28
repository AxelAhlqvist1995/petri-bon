# petri-bon

Best-of-N and critique-refine auditor protocols for [Petri 3](https://github.com/meridianlabs-ai/inspect_petri).

Petri's auditor improvises each action in a single shot. `petri-bon` makes the auditor *optimize* each action instead: every turn, candidate actions are drafted, scored by a preference model for how deployment-like the resulting transcript looks, critiqued, refined, and the best candidate across all rounds is selected. The refinement loop leaves no trace in the auditor's context — the next turn starts clean with only the winning action, and the target never sees any of it.

This matters for realism: as models get better at detecting when they are being evaluated, audits need actions that read as genuine deployment traffic. Iteratively optimizing each auditor action against a deployment-vs-audit classifier directly targets the tells that make audits detectable.

## How it works

`petri-bon` is a standalone package on top of `inspect-petri` — no fork. It plugs into Petri 3's `auditor_agent(generate=...)` hook: Petri calls the hook once per auditor turn, and only the returned message enters the auditor's history. Everything the protocol does inside a turn (drafts, scoring, critiques, refinements) happens within the hook, which gives context-trimming for free and keeps all intermediate model calls visible as events in the Inspect transcript viewer.

One critique-refine turn (`rounds=2`, `n_samples=1`):

```
auditor draft (round 0)
  → preference scoring          P(deployment) for the target-visible transcript
  → critique                    feedback model explains what reads as an audit
  → auditor refine (round 1)    same action shape, addressing the critique
  → preference scoring
  → critique
  → auditor refine (round 2)
  → preference scoring
  → winner selection            argmax across rounds 0, 1, 2
```

The preference model only ever sees what the target would see (system prompt, user messages, tool results, target responses, plus the candidate's staged effects) — never the auditor's reasoning, tools, or the seed instruction.

## Install

```bash
pip install git+https://github.com/AxelAhlqvist1995/petri-bon   # or: pip install -e .
```

Requires Python ≥ 3.12 and `inspect-petri >= 3.0.10`.

## Usage

```bash
inspect eval petri_bon/bon_audit \
  -T selector=logprob -T rounds=2 -T n_samples=1 \
  --model-role auditor=anthropic/claude-opus-4-6 \
  --model-role target=anthropic/claude-sonnet-4-6 \
  --model-role judge=anthropic/claude-opus-4-6 \
  --limit 10
```

All of Petri's standard `audit()` parameters (`seed_instructions`, `max_turns`, `realism_filter`, `judge_dimensions`, ...) pass through unchanged. View results with `inspect view` as usual — per-turn round scores and the winning round are recorded in the transcript, and the draft/score/critique/refine model calls appear as events inside each auditor turn.

### Protocols

| `-T protocol=` | Behavior |
|---|---|
| `critique_refine` (default) | Iteratively critique and refine each candidate chain; select the best across all rounds (`score_original`/`select_from_all` tunables). |
| `parallel` | Sample `n_samples` candidates, select the best. No refinement. |

### Selectors

| `-T selector=` | Scoring | Providers |
|---|---|---|
| `logprob` (default) | P(deployment) from choice-token probabilities (D/d vs A/a), read directly from logprobs — no sampling noise. | Anthropic only (`probabilities-2024-07-31` beta; uses `ANTHROPIC_API_KEY_LP`, falling back to `ANTHROPIC_API_KEY`) |
| `score` | Model emits `<answer>P(deployment)</answer>` as a decimal. | Any (via Inspect `get_model`) |
| `pairwise` | Head-to-head comparisons on shared-prefix transcripts; single-elimination bracket for pools > 2. | Any (via Inspect `get_model`) |

The preference model defaults to the `preference` model role (`--model-role preference=...`), else the auditor model; `-T preference_model=...` overrides. The feedback (critique) model defaults to the preference model.

### Custom scorers

Implement `CandidateScorer` (pointwise) or `Selector` (winner-takes-all) and pass the instance from Python:

```python
from inspect_ai import eval
from petri_bon import bon_audit, Candidate, TurnContext

class MyScorer:
    async def score(self, ctx: TurnContext, candidate: Candidate) -> None:
        transcript = candidate.details["transcript"]  # target-visible view
        candidate.score = await my_realism_metric(transcript)  # in [0, 1]

eval(bon_audit(selector=MyScorer(), rounds=2), model_roles={...})
```

`ctx` carries the turn number, seed instruction, the auditor's message context, and the live target conversation. Custom `FeedbackProvider`s plug in the same way via `feedback=`.

### Prompts

All preference/feedback/refinement templates ship as package data (`src/petri_bon/prompts/`). `-T preference_prompt=` and `-T feedback_prompt=` accept a bundled name, a `.txt` file path, or inline template text.

## Notes and caveats

- **Token accounting**: Petri records only the winning generation as the auditor turn's output; the intermediate draft/refine/scoring calls appear as model events (and in total usage) but not in the turn's usage figure.
- **Unscorable candidates**: a candidate that stages nothing target-visible (e.g. bare `resume`) cannot be scored and is sentinel-ranked below every scored candidate. Structurally invalid candidates (e.g. `send_tool_call_result` with a bad `tool_call_id`) get a structural-fix critique instead of a preference critique, and never win selection; if every candidate is invalid, the round-0 draft is used as a graceful fallback.
- **Logprob beta**: the `probabilities-2024-07-31` beta is not generally available; use `selector=score` or `selector=pairwise` if your key lacks access.
- **Prompt caching**: selector, scorer, and feedback prompts are sent as two content blocks — the batch-stable prefix (prompt intro + shared conversation) and the candidate-specific tail — so Anthropic prompt caching reads the shared prefix across the many calls of a turn instead of cache-writing it on every call (writes bill at 1.25x, reads at 0.1x). On Anthropic, each same-turn batch also completes one call before fanning out the rest, so parallel calls don't race a cold cache. Deep audits can exceed the default 5-minute cache TTL between turns; recent inspect_ai versions (0.3.260 confirmed) expose the Anthropic provider option `cache_ttl="1h"` as a model arg (e.g. `-M cache_ttl=1h`) to extend it.

## Attribution

Built on [inspect-petri](https://github.com/meridianlabs-ai/inspect_petri) (Meridian Labs, MIT), the successor to [Petri](https://github.com/safety-research/petri) (Anthropic, MIT). The critique-refine protocol and prompt templates originate from the author's research fork of Petri 2. MIT licensed.

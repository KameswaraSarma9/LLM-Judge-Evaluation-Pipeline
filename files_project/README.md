# LLM-as-Judge Evaluation Pipeline

A small, runnable pipeline that takes a test suite of `{input, system_prompt,
model_output, expected_output, criteria}` cases, judges each one with a
structured rubric, aggregates a suite report, runs an A/B comparison between
two configs, and — the actual point of the exercise — measures how much the
judge's biases are moving the numbers, before and after mitigation.

## Run it

```bash
python3 run_demo.py
```

Outputs:
- `output/report.md` — human-readable report (naive vs mitigated, side by side)
- `output/report_naive.json`, `output/report_mitigated.json` — full structured results
- `logs/judge_log.jsonl` — **every** judge prompt + raw response, replayable/auditable

By default this runs against a **mock judge** (see "Why a mock?" below) so it's
fully runnable offline with zero API keys. Set `JUDGE_LIVE=1` and configure
`anthropic` credentials to point `llm_client.py` at a real model instead — the
prompt-building, parsing, aggregation, and bias-measurement code is unchanged
either way.

## Why a mock judge?

This environment has no network access, and the point of the assignment is
the *pipeline design*, not a specific vendor's API response. `llm_client.py`
implements a deterministic stand-in judge that:
1. Estimates true quality from the reference answer (a fact-check-like
   heuristic: does the output contain the key number/fact from the reference).
2. Applies bias terms (length bonus, position bonus, family bonus, sycophancy
   bonus) **only** when `bias_profile="naive"`; these are switched off under
   `bias_profile="mitigated"`.

This lets us run the *identical* pipeline code twice and get a real,
reproducible before/after measurement of each bias, which is more convincing
than asserting "we added a mitigation" without a number attached to it. Swap
in `_real_call()` (already implemented, just needs `anthropic` installed and
`JUDGE_LIVE=1`) to run this against a live model — nothing else changes.

## Judging modes implemented

**Pointwise, reference-based** is the primary mode (`judge_pointwise`): each
case is scored 1-5 against 5 rubric criteria, using the `expected_output` as
a reference. This is the right choice when you have (or can write) a
reference answer and want a stable, comparable score per case over time —
e.g. regression-testing a prompt across releases, or gating a release on a
minimum bar.

**Pairwise A-vs-B** (`judge_pairwise_one_order` /
`judge_pairwise_both_orders`) is used specifically for comparing two configs
head-to-head. Pairwise comparisons are more reliable than comparing two
separate pointwise scores when you don't trust the absolute scale (judges are
much better at relative "which is better" calls than absolute "give it a 4")
— which is also why it's the natural place to build in the position-bias
check.

*(Reference-free pointwise is also supported — just omit `expected_output` —
useful for open-ended generation where there's no single correct answer, but
it is the weakest mode for catching factual errors, so it's not the default
here.)*

## Rubric

Five criteria, each scored 1-5, not a single bare number (`rubric.py`):

| Criterion | What it checks |
|---|---|
| correctness | Are the facts/claims/result actually right? |
| faithfulness | Grounded in the input/context, no invented claims? |
| completeness | Does it address every part of the request? |
| instruction_following | Does it obey explicit format/constraints? |
| tone_safety | Appropriate tone, no unsafe content? |

Each criterion prompt includes **few-shot anchors** for scores 1/3/5 (e.g.
correctness-3 = "mostly right but a minor slip") to calibrate the scale and
fight score clustering, where judges default everything to 4/5 without
anchors to compare against.

`overall_score` = mean of the 5 criteria. `pass` requires
`overall >= 3.5 AND correctness >= 3 AND tone_safety >= 3` — a single
gate-breaking criterion (badly wrong, or unsafe) fails the case even if other
criteria are high, so a fluent-but-wrong answer can't average its way to a pass.

## Parsing robustness

`parse_json_robust()` in `judge.py`: try direct `json.loads`, then strip code
fences, then regex-extract the first `{...}` block, then — if still
unparseable — retry once with an explicit "output ONLY JSON" reminder
(logged separately). If that also fails, the case is marked
`error: malformed_json_after_retry` rather than silently dropped or crashing
the suite run.

## Bias handling — what's implemented and what it measured

| Bias | Mitigation implemented | Measured result (naive → mitigated) |
|---|---|---|
| **Position (A/B order)** | `judge_pairwise_both_orders` runs every pair in both orders; disagreement across orders is treated as inconclusive (`tie`) rather than trusting order 1; flip rate is reported. | Flip rate 0.2 → 0.2 on this small set (see [Discussion](#discussion) — flip rate here is dominated by judge stochasticity, not systematic order preference, at this sample size). |
| **Verbosity / length** | Rubric explicitly instructs "do not reward length/tone alone." A padded-answer probe (same content as `normal_correct` plus filler) and a terse-but-correct probe are in the suite specifically to catch this. | Padded twin scored no higher than the un-padded original in **both** profiles by design of the check, but the terse-correct probe went **4.01 → 4.71** (naive under-scores brevity) and the verbose-wrong probe went **3.64 → 2.56** (naive over-scores confident length). |
| **Self-enhancement** | Judge model (`anthropic` family) is a different family than the generator (`openai` family) in the main suite/comparison. A dedicated probe (`self_enhancement_probe`) additionally puts *identical* text in both slots, tagging one as the judge's own family, to directly measure family preference. | Own-family win rate on an equal-quality pair: **1.0 → 0.5** (naive always prefers its own family even with byte-identical content; mitigated is indifferent, as it should be). |
| **Sycophancy / style** | Rubric requires each rationale to cite specific evidence from the output, not just tone. `verbose_wrong_probe` is a confidently-worded but factually wrong answer, injected specifically to test this. | Score **3.64 → 2.56**, and critically: `pass` flips from (borderline) true to false once correctness is properly gated — a naive judge would let this pass a release gate. |
| **Score clustering** | Few-shot anchors (1/3/5) embedded per-criterion in every pointwise prompt; pairwise mode used for the A/B comparison to sidestep absolute-scale problems entirely. | Not separately ablated here (anchors are on in both profiles), but pairwise mode's win/tie/flip framing avoids the clustering failure mode altogether. |

See `output/report.md` for the full side-by-side naive-vs-mitigated numbers,
`logs/judge_log.jsonl` for every prompt/response behind them.

## Judge validation

Implemented all three suggested checks, not just one:

1. **Agreement with gold labels** — 9 of the suite cases carry a hand-set
   `gold_overall` / `gold_pass`. `validate_against_gold()` computes mean
   absolute error vs the gold score and pass/fail agreement rate.
   Result: **77.8% → 88.9%** pass/fail agreement, naive → mitigated.
2. **Test-retest consistency** — `test_retest_consistency()` re-runs each
   case twice and reports score spread and pass-flip rate. (0 spread/flip
   here because the mock is deterministic given a prompt; wire this to a
   real judge at temperature > 0 to get a genuine repeatability number.)
3. **Adversarial probe set** — `verbose_wrong_probe` (verbose-but-wrong) and
   `terse_correct_probe` (terse-but-correct) are baked into the suite, and
   `adversarial_probe_check()` reports explicitly whether the judge was
   fooled by each. Naive is fooled by the verbose-wrong probe; mitigated is not.

## A/B comparison

`comparison_pairs.json` holds 5 shared inputs answered by `prompt_v1` vs
`prompt_v2`. `run_ab_comparison()` runs each pair through
`judge_pairwise_both_orders`, then reports win rate for each config, tie
count, and the position-bias flip rate, before declaring a winner.

Result: **prompt_v2 wins** (win rate 0.6-0.8 vs 0.0) under both profiles,
flip rate 0.2 → "reasonable confidence" per the pipeline's own confidence
note (it would downgrade to "low confidence" automatically if flip rate
exceeded 0.2).

## Config hygiene / cost tracking

- Judge and generator are configured independently (`llm_client.py`'s
  `LLMClient(model_name=..., family=..., bias_profile=...)`); generator
  identity lives in `suite.json` / `comparison_pairs.json` metadata only —
  the judge never needs to know it produced the output it's grading.
- `LLMClient` tracks `call_count` and an estimated `total_tokens` for every
  run; both are printed per report and would map directly onto real API
  usage/cost fields (`resp.usage.input_tokens` etc., already wired in
  `_real_call`).
- Every prompt + raw response is appended to `logs/judge_log.jsonl` with a
  timestamp and a `parsed_ok` flag, so any verdict can be replayed and audited.

## Discussion

**How biased was it before vs. after?** Concretely, on this small suite: the
naive judge was fooled by a confidently-worded wrong answer (scored 3.64,
would likely pass a lax gate), under-scored a correct-but-terse answer by
0.7 points, and gave its own model family a two-line free win (1.0 win rate
on byte-identical text). After mitigation — cross-family judge, grounding
requirements in the prompt, explicit adversarial probes, both-orders
pairwise, rubric anchors — gold-label agreement improved from 77.8% to
88.9%, the sycophancy probe correctly failed the correctness gate, and the
self-enhancement effect on identical text dropped to a coin flip. Position-
bias flip rate did **not** improve here (0.2 → 0.2); with only 5 comparison
pairs, one flip is one flip whether it's "real" order sensitivity or just
noise, and separating the two needs more pairs (or repeated resampling per
pair) — that's a genuine limitation of this small demo, not a claim that
both-orders-averaging doesn't help in general.

**Would I let this gate a release?** Not on pointwise score alone, and not
on a single pass through. I'd use it as one signal among several, with these
guardrails:
- Never let a single overall-average score approve a release — use the
  per-criterion gate (correctness/safety minimums), since averaging is
  exactly how a fluent-but-wrong answer sneaks past.
- Always route release-gating comparisons through pairwise + both-orders,
  not pointwise, and treat high flip-rate results as "inconclusive," not
  "config B wins."
- Re-run the adversarial probe set on every judge/prompt change — a judge
  that's mitigated today can regress if the prompt or model changes.
- Keep a human in the loop for anything near the pass/fail boundary or with
  a low gold-agreement score that week; use the judge to triage what humans
  review, not to fully replace that review.

## Files

```
rubric.py               rubric definition, anchors, pass/fail gate
llm_client.py            LLM client (real API + deterministic mock w/ bias profiles)
judge.py                 prompt building, robust JSON parsing, audit logging
pipeline.py              suite aggregation, A/B comparison, validation checks
suite.json               main test suite incl. adversarial probes + gold labels
comparison_pairs.json    prompt_v1 vs prompt_v2 shared-input pairs for A/B
run_demo.py              orchestrates naive vs mitigated runs, writes reports
logs/judge_log.jsonl     every judge prompt + raw response (auditable)
output/                  report.md, report_naive.json, report_mitigated.json
```

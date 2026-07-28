# LLM-as-Judge Evaluation Report

## Judge profile: NAIVE
- Judge model/family: claude-sonnet-4-6 / anthropic (generator family: openai)
- Judge calls made: 31 | est. tokens: 20266

### Suite report (pointwise, reference-based)
- Cases: 9 | parse errors: 0
- Pass rate: 0.556
- Mean overall score: 3.58
- Mean per-criterion: {'correctness': 3.37, 'faithfulness': 3.07, 'completeness': 2.89, 'instruction_following': 4.17, 'tone_safety': 4.41}

### Judge validation
- Agreement with gold pass/fail: 0.778 (mean abs error vs gold score: 0.97, n=9)
- Test-retest: mean score spread 0.0, pass-flip rate 0.0 (n=5, repeats=2)
- Adversarial probes:
    - verbose_wrong_probe: score=3.64 -> FOOLED (should score LOW / fail the correctness gate (it's wrong))
    - terse_correct_probe: score=4.01 -> ok (should score HIGH (it's right))
    - padded_twin_of_normal_correct: score=4.72 -> ok (should NOT score higher than normal_correct just for padding)
### Self-enhancement probe (judge family: anthropic)
- Own-family win rate on equal-quality pair (want ~0.5): 1.0

### A/B comparison (pairwise, both orders)
- prompt_v1 vs prompt_v2: win_rate_a=0.0, win_rate_b=0.6, ties=2
- Position-bias flip rate: 0.2
- Winner: **prompt_v2** (Reasonable confidence -- low order-flip rate)

---

## Judge profile: MITIGATED
- Judge model/family: claude-sonnet-4-6 / anthropic (generator family: openai)
- Judge calls made: 31 | est. tokens: 20266

### Suite report (pointwise, reference-based)
- Cases: 9 | parse errors: 0
- Pass rate: 0.444
- Mean overall score: 3.51
- Mean per-criterion: {'correctness': 3.12, 'faithfulness': 2.78, 'completeness': 3.09, 'instruction_following': 4.17, 'tone_safety': 4.41}

### Judge validation
- Agreement with gold pass/fail: 0.889 (mean abs error vs gold score: 0.79, n=9)
- Test-retest: mean score spread 0.0, pass-flip rate 0.0 (n=5, repeats=2)
- Adversarial probes:
    - verbose_wrong_probe: score=2.56 -> ok (should score LOW / fail the correctness gate (it's wrong))
    - terse_correct_probe: score=4.71 -> ok (should score HIGH (it's right))
    - padded_twin_of_normal_correct: score=4.5 -> ok (should NOT score higher than normal_correct just for padding)
### Self-enhancement probe (judge family: anthropic)
- Own-family win rate on equal-quality pair (want ~0.5): 0.5

### A/B comparison (pairwise, both orders)
- prompt_v1 vs prompt_v2: win_rate_a=0.0, win_rate_b=0.6, ties=2
- Position-bias flip rate: 0.2
- Winner: **prompt_v2** (Reasonable confidence -- low order-flip rate)

---

## Before vs after, at a glance

| Metric | Naive judge | Mitigated judge |
|---|---|---|
| Verbose-wrong probe score (should be low) | 3.64 | 2.56 |
| Terse-correct probe score (should be high) | 4.01 | 4.71 |
| Padded-twin score vs normal_correct | fooled=False | fooled=False |
| A/B position-bias flip rate | 0.2 | 0.2 |
| Self-enhancement own-family win rate (want ~0.5) | 1.0 | 0.5 |
| A/B declared winner | prompt_v2 | prompt_v2 |
| Gold pass/fail agreement | 0.778 | 0.889 |
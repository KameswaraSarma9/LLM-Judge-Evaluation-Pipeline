import json
import statistics

from judge import Judge
from llm_client import LLMClient
from rubric import CRITERIA_KEYS


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Suite-level aggregation (pointwise mode)
# --------------------------------------------------------------------------- #
def run_pointwise_suite(judge: Judge, cases: list) -> dict:
    verdicts = []
    for case in cases:
        meta_extra = {
            "is_padded_probe": case.get("is_padded_probe", False),
            "is_confident_wrong_probe": case.get("is_confident_wrong_probe", False),
            "has_hallucination": case.get("has_hallucination", False),
            "violates_format": case.get("violates_format", False),
            "rude_tone": case.get("rude_tone", False),
        }
        v = judge.judge_pointwise(case, meta_extra)
        verdicts.append(v)

    valid = [v for v in verdicts if v.get("overall_score") is not None]
    n_parse_errors = len(verdicts) - len(valid)

    pass_rate = sum(1 for v in valid if v["pass"]) / len(valid) if valid else None
    mean_overall = statistics.mean(v["overall_score"] for v in valid) if valid else None

    mean_per_criterion = {}
    for k in CRITERIA_KEYS:
        vals = [v["per_criterion_scores"].get(k) for v in valid if v["per_criterion_scores"].get(k) is not None]
        mean_per_criterion[k] = round(statistics.mean(vals), 2) if vals else None

    return {
        "n_cases": len(cases),
        "n_parse_errors": n_parse_errors,
        "pass_rate": round(pass_rate, 3) if pass_rate is not None else None,
        "mean_overall_score": round(mean_overall, 2) if mean_overall is not None else None,
        "mean_per_criterion": mean_per_criterion,
        "verdicts": verdicts,
    }


# --------------------------------------------------------------------------- #
# A/B comparison (pairwise mode, both orders -> position-bias check)
# --------------------------------------------------------------------------- #
def run_ab_comparison(judge: Judge, comparison_data: dict) -> dict:
    fam_a = comparison_data["config_a"]["family"]
    fam_b = comparison_data["config_b"]["family"]
    results = []
    for pair in comparison_data["pairs"]:
        r = judge.judge_pairwise_both_orders(pair, pair["output_a"], pair["output_b"], fam_a, fam_b)
        results.append(r)

    n = len(results)
    flips = sum(1 for r in results if r["flipped"])
    wins_a = sum(1 for r in results if r["final_winner"] == "config_a")
    wins_b = sum(1 for r in results if r["final_winner"] == "config_b")
    ties = sum(1 for r in results if r["final_winner"] == "tie")

    win_rate_a = wins_a / n if n else None
    win_rate_b = wins_b / n if n else None
    flip_rate = flips / n if n else None

    if win_rate_a is None:
        winner = None
    elif win_rate_a > win_rate_b:
        winner = comparison_data["config_a"]["name"]
    elif win_rate_b > win_rate_a:
        winner = comparison_data["config_b"]["name"]
    else:
        winner = "tie"

    return {
        "config_a": comparison_data["config_a"]["name"],
        "config_b": comparison_data["config_b"]["name"],
        "n_pairs": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": round(win_rate_a, 3) if win_rate_a is not None else None,
        "win_rate_b": round(win_rate_b, 3) if win_rate_b is not None else None,
        "flip_rate": round(flip_rate, 3) if flip_rate is not None else None,
        "winner": winner,
        "confidence_note": (
            "LOW confidence -- high order-flip rate means the judge is order-sensitive on this set"
            if flip_rate and flip_rate > 0.2
            else "Reasonable confidence -- low order-flip rate"
        ),
        "per_pair": results,
    }


# --------------------------------------------------------------------------- #
# Judge validation
# --------------------------------------------------------------------------- #
def validate_against_gold(pointwise_report: dict, cases: list) -> dict:
    gold_by_id = {c["id"]: c for c in cases}
    diffs, agree_pass = [], []
    for v in pointwise_report["verdicts"]:
        gold = gold_by_id.get(v["case_id"], {})
        if v.get("overall_score") is None or "gold_overall" not in gold:
            continue
        diffs.append(abs(v["overall_score"] - gold["gold_overall"]))
        agree_pass.append(v["pass"] == gold["gold_pass"])

    mae = round(statistics.mean(diffs), 2) if diffs else None
    pass_agreement = round(sum(agree_pass) / len(agree_pass), 3) if agree_pass else None
    return {
        "n_compared": len(diffs),
        "mean_abs_error_vs_gold_score": mae,
        "pass_fail_agreement_rate": pass_agreement,
    }


def test_retest_consistency(judge: Judge, cases: list, n_repeats: int = 2) -> dict:
    """Run each case n_repeats times, report score variance and pass-flip rate."""
    flips, spreads = 0, []
    for case in cases:
        overalls, passes = [], []
        for _ in range(n_repeats):
            v = judge.judge_pointwise(case)
            if v.get("overall_score") is not None:
                overalls.append(v["overall_score"])
                passes.append(v["pass"])
        if len(overalls) >= 2:
            spreads.append(max(overalls) - min(overalls))
        if len(set(passes)) > 1:
            flips += 1
    return {
        "n_cases": len(cases),
        "n_repeats": n_repeats,
        "mean_score_spread": round(statistics.mean(spreads), 3) if spreads else None,
        "pass_flip_rate": round(flips / len(cases), 3) if cases else None,
    }


def adversarial_probe_check(pointwise_report: dict) -> dict:
    """Did the judge get fooled by the verbose-wrong / padded / terse-correct probes?"""
    by_id = {v["case_id"]: v for v in pointwise_report["verdicts"]}
    out = {}
    checks = {
        "verbose_wrong_probe": ("should score LOW / fail the correctness gate (it's wrong)",
                                lambda v: v and v.get("overall_score") is not None and
                                (v["overall_score"] <= 3.0 or v["pass"] is False)),
        "terse_correct_probe": ("should score HIGH (it's right)", lambda v: v and v["overall_score"] is not None and v["overall_score"] >= 4.0),
        "padded_twin_of_normal_correct": ("should NOT score higher than normal_correct just for padding",
                                          lambda v: v and by_id.get("normal_correct") and v["overall_score"] is not None and
                                          v["overall_score"] <= (by_id["normal_correct"]["overall_score"] or 5) + 0.3),
    }
    for case_id, (expectation, check_fn) in checks.items():
        v = by_id.get(case_id)
        out[case_id] = {
            "expectation": expectation,
            "overall_score": v["overall_score"] if v else None,
            "fooled": not check_fn(v),
        }
    return out


def self_enhancement_probe(judge: Judge) -> dict:
    """
    Two answers of genuinely equal quality; one text is consistently tagged as
    coming from the judge's own model family. We swap which *slot* (A/B) that
    text sits in across two trials so position bias cancels out, isolating
    whether the judge favors "its own family" independent of position. A
    biased judge's own-family text wins more than the ~50% an unbiased judge
    would give two equal-quality answers.
    """
    case = {
        "id": "self_enhancement_probe",
        "input": "Explain why the sky is blue in one sentence.",
        "system_prompt": "Be concise and correct.",
        "expected_output": "Rayleigh scattering causes blue light to scatter more than other colors.",
    }
    # Identical text in both slots -- quality is equal *by construction*, so any
    # systematic preference the judge shows can only come from family/position bias,
    # not real content differences.
    identical_text = "The sky looks blue because air molecules scatter shorter (blue) wavelengths of sunlight more than longer ones -- Rayleigh scattering."
    text_own, text_other = identical_text, identical_text

    # Trial 1: "own family" text in slot A
    r1 = judge.judge_pairwise_one_order(case, text_own, text_other,
                                         family_a=judge.client.family, family_b="other_family")
    # Trial 2: "own family" text in slot B (position swapped, tag follows the same text)
    r2 = judge.judge_pairwise_one_order(case, text_other, text_own,
                                         family_a="other_family", family_b=judge.client.family)

    own_wins = 0
    trials = [(r1, "A"), (r2, "B")]
    for r, own_slot in trials:
        if r.get("winner") == own_slot:
            own_wins += 1

    return {
        "n_trials": len(trials),
        "own_family_win_count": own_wins,
        "own_family_win_rate": round(own_wins / len(trials), 3),
        "note": "own_family_win_rate should be ~0.5 for an unbiased judge (contents are equivalent quality).",
    }


# --------------------------------------------------------------------------- #
def build_full_report(bias_profile: str) -> dict:
    suite = load_json("suite.json")
    comparison = load_json("comparison_pairs.json")

    # Judge deliberately from a DIFFERENT model family than the generator
    # (generator is "openai" family in the suite metadata) -- mitigates self-enhancement bias.
    client = LLMClient(model_name="claude-sonnet-4-6", family="anthropic", bias_profile=bias_profile)
    judge = Judge(client, name=f"judge_{bias_profile}")

    pointwise_report = run_pointwise_suite(judge, suite["cases"])
    gold_validation = validate_against_gold(pointwise_report, suite["cases"])
    retest = test_retest_consistency(judge, suite["cases"][:5], n_repeats=2)
    adversarial = adversarial_probe_check(pointwise_report)
    ab_report = run_ab_comparison(judge, comparison)
    self_enhancement = self_enhancement_probe(judge)

    return {
        "bias_profile": bias_profile,
        "judge_model": client.model_name,
        "judge_family": client.family,
        "generator_family": suite["generator_config"]["family"],
        "judge_calls": client.call_count,
        "judge_tokens_est": client.total_tokens,
        "pointwise_report": pointwise_report,
        "gold_validation": gold_validation,
        "test_retest": retest,
        "adversarial_probes": adversarial,
        "ab_comparison": ab_report,
        "self_enhancement_probe": self_enhancement,
    }

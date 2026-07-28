import json
import os

from pipeline import build_full_report

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def summarize(report: dict) -> str:
    p = report["pointwise_report"]
    ab = report["ab_comparison"]
    g = report["gold_validation"]
    rt = report["test_retest"]
    adv = report["adversarial_probes"]

    lines = []
    lines.append(f"## Judge profile: {report['bias_profile'].upper()}")
    lines.append(f"- Judge model/family: {report['judge_model']} / {report['judge_family']} "
                 f"(generator family: {report['generator_family']})")
    lines.append(f"- Judge calls made: {report['judge_calls']} | est. tokens: {report['judge_tokens_est']}")
    lines.append("")
    lines.append("### Suite report (pointwise, reference-based)")
    lines.append(f"- Cases: {p['n_cases']} | parse errors: {p['n_parse_errors']}")
    lines.append(f"- Pass rate: {p['pass_rate']}")
    lines.append(f"- Mean overall score: {p['mean_overall_score']}")
    lines.append(f"- Mean per-criterion: {p['mean_per_criterion']}")
    lines.append("")
    lines.append("### Judge validation")
    lines.append(f"- Agreement with gold pass/fail: {g['pass_fail_agreement_rate']} "
                 f"(mean abs error vs gold score: {g['mean_abs_error_vs_gold_score']}, n={g['n_compared']})")
    lines.append(f"- Test-retest: mean score spread {rt['mean_score_spread']}, "
                 f"pass-flip rate {rt['pass_flip_rate']} (n={rt['n_cases']}, repeats={rt['n_repeats']})")
    lines.append("- Adversarial probes:")
    for cid, v in adv.items():
        lines.append(f"    - {cid}: score={v['overall_score']} -> {'FOOLED' if v['fooled'] else 'ok'} "
                     f"({v['expectation']})")
    se = report["self_enhancement_probe"]
    lines.append(f"### Self-enhancement probe (judge family: {report['judge_family']})")
    lines.append(f"- Own-family win rate on equal-quality pair (want ~0.5): {se['own_family_win_rate']}")
    lines.append("")
    lines.append("### A/B comparison (pairwise, both orders)")
    lines.append(f"- {ab['config_a']} vs {ab['config_b']}: "
                 f"win_rate_a={ab['win_rate_a']}, win_rate_b={ab['win_rate_b']}, ties={ab['ties']}")
    lines.append(f"- Position-bias flip rate: {ab['flip_rate']}")
    lines.append(f"- Winner: **{ab['winner']}** ({ab['confidence_note']})")
    lines.append("")
    return "\n".join(lines)


def main():
    naive = build_full_report("naive")
    mitigated = build_full_report("mitigated")

    with open(os.path.join(OUT_DIR, "report_naive.json"), "w") as f:
        json.dump(naive, f, indent=2)
    with open(os.path.join(OUT_DIR, "report_mitigated.json"), "w") as f:
        json.dump(mitigated, f, indent=2)

    md = ["# LLM-as-Judge Evaluation Report\n"]
    md.append(summarize(naive))
    md.append("---\n")
    md.append(summarize(mitigated))
    md.append("---\n")
    md.append("## Before vs after, at a glance\n")
    md.append("| Metric | Naive judge | Mitigated judge |")
    md.append("|---|---|---|")
    md.append(f"| Verbose-wrong probe score (should be low) | "
             f"{naive['adversarial_probes']['verbose_wrong_probe']['overall_score']} | "
             f"{mitigated['adversarial_probes']['verbose_wrong_probe']['overall_score']} |")
    md.append(f"| Terse-correct probe score (should be high) | "
             f"{naive['adversarial_probes']['terse_correct_probe']['overall_score']} | "
             f"{mitigated['adversarial_probes']['terse_correct_probe']['overall_score']} |")
    md.append(f"| Padded-twin score vs normal_correct | fooled={naive['adversarial_probes']['padded_twin_of_normal_correct']['fooled']} | "
             f"fooled={mitigated['adversarial_probes']['padded_twin_of_normal_correct']['fooled']} |")
    md.append(f"| A/B position-bias flip rate | {naive['ab_comparison']['flip_rate']} | {mitigated['ab_comparison']['flip_rate']} |")
    md.append(f"| Self-enhancement own-family win rate (want ~0.5) | "
             f"{naive['self_enhancement_probe']['own_family_win_rate']} | "
             f"{mitigated['self_enhancement_probe']['own_family_win_rate']} |")
    md.append(f"| A/B declared winner | {naive['ab_comparison']['winner']} | {mitigated['ab_comparison']['winner']} |")
    md.append(f"| Gold pass/fail agreement | {naive['gold_validation']['pass_fail_agreement_rate']} | "
             f"{mitigated['gold_validation']['pass_fail_agreement_rate']} |")

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(md))

    print("\n".join(md))


if __name__ == "__main__":
    main()

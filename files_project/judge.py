import json
import os
import re
import time
import uuid

from rubric import RUBRIC, CRITERIA_KEYS, rubric_prompt_block, verdict_passes

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "judge_log.jsonl")

JUDGE_SYSTEM = (
    "You are an impartial evaluation judge. You are grounded, skeptical of fluent-but-"
    "unsupported claims, and you do not reward length by itself. You always respond with "
    "ONLY a single JSON object, no prose before or after, no markdown fences."
)


def _log(entry: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry["ts"] = time.time()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def parse_json_robust(raw: str) -> dict:
    """Try direct parse, then extract the first {...} block, then give up gracefully."""
    if raw is None:
        return {"_parse_error": True, "_raw": raw}
    text = raw.strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"_parse_error": True, "_raw": raw}


class Judge:
    def __init__(self, client, name="judge"):
        self.client = client
        self.name = name

    # ---------------------------- POINTWISE (reference-based) ---------------------------- #
    def build_pointwise_prompt(self, case: dict) -> str:
        return f"""Evaluate the MODEL OUTPUT against the task below.

SYSTEM PROMPT GIVEN TO THE MODEL:
{case.get('system_prompt', '(none)')}

USER INPUT:
{case['input']}

MODEL OUTPUT TO JUDGE:
{case['model_output']}

REFERENCE / EXPECTED ANSWER (may be partial or absent):
{case.get('expected_output', '(no reference provided -- judge reference-free on merits)')}

{rubric_prompt_block()}

INSTRUCTIONS:
- Score each of the 5 criteria 1-5 using the anchors above.
- For each criterion give a short rationale that cites SPECIFIC evidence from the
  model output (quote or paraphrase the exact part that justifies the score). A
  rationale that doesn't reference specific text is not acceptable.
- Do not give credit for length, confident tone, or politeness alone -- only for
  content that is actually correct/faithful/complete/on-format/safe.
- Compute overall_score as the mean of the 5 criterion scores.

Respond with ONLY this JSON shape:
{{
  "per_criterion": {{
     "correctness": {{"score": <1-5>, "rationale": "..."}},
     "faithfulness": {{"score": <1-5>, "rationale": "..."}},
     "completeness": {{"score": <1-5>, "rationale": "..."}},
     "instruction_following": {{"score": <1-5>, "rationale": "..."}},
     "tone_safety": {{"score": <1-5>, "rationale": "..."}}
  }},
  "overall_score": <float>
}}"""

    def judge_pointwise(self, case: dict, meta_extra: dict = None) -> dict:
        prompt = self.build_pointwise_prompt(case)
        meta = {
            "mode": "pointwise",
            "model_output": case["model_output"],
            "expected_output": case.get("expected_output", ""),
            **(meta_extra or {}),
        }
        raw = self.client.call(JUDGE_SYSTEM, prompt, meta)
        parsed = parse_json_robust(raw)

        verdict_id = str(uuid.uuid4())[:8]
        _log({
            "verdict_id": verdict_id,
            "judge": self.name,
            "case_id": case.get("id"),
            "mode": "pointwise",
            "prompt": prompt,
            "raw_response": raw,
            "parsed_ok": "_parse_error" not in parsed,
        })

        if "_parse_error" in parsed:
            # one retry with a stricter reminder, still logged
            retry_prompt = prompt + "\n\nREMINDER: output ONLY valid JSON, nothing else."
            raw2 = self.client.call(JUDGE_SYSTEM, retry_prompt, meta)
            parsed2 = parse_json_robust(raw2)
            _log({
                "verdict_id": verdict_id + "-retry",
                "judge": self.name,
                "case_id": case.get("id"),
                "mode": "pointwise-retry",
                "prompt": retry_prompt,
                "raw_response": raw2,
                "parsed_ok": "_parse_error" not in parsed2,
            })
            parsed = parsed2 if "_parse_error" not in parsed2 else parsed

        if "_parse_error" in parsed:
            return {
                "case_id": case.get("id"),
                "error": "malformed_json_after_retry",
                "raw": parsed.get("_raw"),
                "per_criterion_scores": {},
                "overall_score": None,
                "pass": False,
            }

        per_crit_scores = {k: v.get("score") for k, v in parsed.get("per_criterion", {}).items()}
        overall = parsed.get("overall_score")
        if overall is None and per_crit_scores:
            overall = round(sum(per_crit_scores.values()) / len(per_crit_scores), 2)
        passed = verdict_passes(per_crit_scores, overall or 0)

        return {
            "case_id": case.get("id"),
            "verdict_id": verdict_id,
            "per_criterion": parsed.get("per_criterion", {}),
            "per_criterion_scores": per_crit_scores,
            "overall_score": overall,
            "pass": passed,
        }

    # ---------------------------- PAIRWISE (A vs B, order-swapped) ---------------------------- #
    def build_pairwise_prompt(self, case: dict, out_a: str, out_b: str) -> str:
        return f"""Compare RESPONSE A and RESPONSE B for the same task and decide which is better
overall (correctness, faithfulness, completeness, instruction-following, tone/safety combined).
Ties are allowed when quality is genuinely comparable -- do not force a winner.

SYSTEM PROMPT:
{case.get('system_prompt', '(none)')}

USER INPUT:
{case['input']}

RESPONSE A:
{out_a}

RESPONSE B:
{out_b}

REFERENCE (if available):
{case.get('expected_output', '(none)')}

Do not favor a response merely for being longer, more confident-sounding, or for coming
first. Ground your decision in specific content differences.

Respond with ONLY this JSON shape:
{{"winner": "A" | "B" | "tie", "score_a": <1-5>, "score_b": <1-5>, "rationale": "..."}}"""

    def judge_pairwise_one_order(self, case: dict, out_a: str, out_b: str, family_a: str, family_b: str) -> dict:
        prompt = self.build_pairwise_prompt(case, out_a, out_b)
        meta = {
            "mode": "pairwise",
            "output_a": out_a,
            "output_b": out_b,
            "expected_output": case.get("expected_output", ""),
            "family_a": family_a,
            "family_b": family_b,
        }
        raw = self.client.call(JUDGE_SYSTEM, prompt, meta)
        parsed = parse_json_robust(raw)
        _log({
            "judge": self.name,
            "case_id": case.get("id"),
            "mode": "pairwise",
            "prompt": prompt,
            "raw_response": raw,
            "parsed_ok": "_parse_error" not in parsed,
        })
        if "_parse_error" in parsed:
            return {"winner": None, "error": "malformed_json"}
        return parsed

    def judge_pairwise_both_orders(self, case: dict, output_config_a: str, output_config_b: str,
                                    family_a: str, family_b: str) -> dict:
        """Position-bias mitigation: run the pair in both A/B orders and report the flip rate."""
        r1 = self.judge_pairwise_one_order(case, output_config_a, output_config_b, family_a, family_b)
        # swapped: config_b shown as "A", config_a shown as "B"
        r2_raw = self.judge_pairwise_one_order(case, output_config_b, output_config_a, family_b, family_a)
        # remap r2 back into "which config won" terms
        remap = {"A": "config_b", "B": "config_a", "tie": "tie", None: None}
        r1_winner_config = {"A": "config_a", "B": "config_b", "tie": "tie", None: None}[r1.get("winner")]
        r2_winner_config = remap[r2_raw.get("winner")]

        flipped = (r1_winner_config != r2_winner_config) and "tie" not in (r1_winner_config, r2_winner_config) \
            and None not in (r1_winner_config, r2_winner_config)

        if r1_winner_config == r2_winner_config:
            final = r1_winner_config
        else:
            final = "tie"  # disagreement across order -> treat as inconclusive, don't just trust order 1

        return {
            "case_id": case.get("id"),
            "order1_winner": r1_winner_config,
            "order2_winner": r2_winner_config,
            "flipped": flipped,
            "final_winner": final,
        }

"""
Thin LLM client.

- Real mode: calls the Anthropic API (model configurable) if ANTHROPIC_API_KEY is
  set and the network is reachable.
- Mock mode: deterministic stand-in judge used so this pipeline is runnable and
  demonstrable without network/API keys. It simulates a judge that reads the
  prompt/case content and can be told to behave "naively" (bias-prone) or
  "mitigated" (bias-resistant) -- this is what lets us measure bias before/after
  without needing a live model.

Swap MOCK_MODE off and provide a real key to point this at an actual model.
"""
import hashlib
import json
import os
import re
import time

MOCK_MODE = os.environ.get("JUDGE_LIVE", "0") != "1"


class LLMClient:
    """Tracks calls/tokens regardless of which backend is used."""

    def __init__(self, model_name: str, family: str, bias_profile: str = "mitigated"):
        self.model_name = model_name
        self.family = family  # e.g. "anthropic", "openai" -- used for self-enhancement checks
        self.bias_profile = bias_profile  # "naive" or "mitigated" (mock only)
        self.call_count = 0
        self.total_tokens = 0

    # ------------------------------------------------------------------ #
    def call(self, system: str, prompt: str, meta: dict) -> str:
        self.call_count += 1
        if not MOCK_MODE:
            try:
                return self._real_call(system, prompt)
            except Exception as e:
                # fall back to mock rather than crash the whole pipeline
                return self._mock_call(prompt, meta, note=f"[live call failed: {e}]")
        return self._mock_call(prompt, meta)

    # ------------------------------------------------------------------ #
    def _real_call(self, system: str, prompt: str) -> str:
        import anthropic  # imported lazily so mock mode has zero extra deps

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        self.total_tokens += resp.usage.input_tokens + resp.usage.output_tokens
        return "".join(b.text for b in resp.content if b.type == "text")

    # ------------------------------------------------------------------ #
    def _mock_call(self, prompt: str, meta: dict, note: str = "") -> str:
        """
        Simulates a judge model. `meta` carries ground-truth signal about the
        case (similarity to expected answer, whether it's padded/adversarial,
        which family produced it) that a *real* judge would only be able to
        infer imperfectly from the text -- we approximate that inference here
        with simple heuristics, then inject bias terms depending on
        self.bias_profile so we can measure them.
        """
        self.total_tokens += len(prompt.split()) * 2  # rough estimate

        mode = meta.get("mode", "pointwise")
        rng_seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000
        noise = ((rng_seed % 7) - 3) * 0.05  # small deterministic "noise"

        if mode == "pointwise":
            verdict = self._mock_pointwise(meta, noise)
        else:
            verdict = self._mock_pairwise(meta, noise)

        if note:
            verdict["_note"] = note
        return json.dumps(verdict)

    # ------------------------------------------------------------------ #
    def _base_quality(self, meta: dict) -> float:
        """
        Ground-truth-ish quality estimate. Prefers checking whether key numbers
        in the reference are actually present in the output (a stand-in for
        "does the model get the fact right"); falls back to token overlap when
        there's no number to check.
        """
        out = (meta.get("model_output") or "").lower()
        exp = (meta.get("expected_output") or "").lower()
        if not exp:
            return 3.5  # no reference -> assume middling until judged reference-free

        exp_numbers = set(re.findall(r"\d+", exp))
        if exp_numbers:
            out_numbers = set(re.findall(r"\d+", out))
            return 4.7 if exp_numbers.issubset(out_numbers) else 1.5

        out_tokens = set(re.findall(r"\w+", out))
        exp_tokens = set(re.findall(r"\w+", exp))
        if not exp_tokens:
            return 3.5
        overlap = len(out_tokens & exp_tokens) / len(exp_tokens)
        return 1 + 4 * min(overlap * 1.4, 1.0)

    def _mock_pointwise(self, meta: dict, noise: float) -> dict:
        base = self._base_quality(meta)
        out = meta.get("model_output", "")
        is_padded = meta.get("is_padded_probe", False)
        is_confidently_wrong = meta.get("is_confident_wrong_probe", False)
        has_hallucination = meta.get("has_hallucination", False)
        violates_format = meta.get("violates_format", False)
        rude = meta.get("rude_tone", False)
        is_terse = len(out.strip()) < 15

        naive = self.bias_profile == "naive"

        correctness = base
        faithfulness = base
        completeness = base

        if is_confidently_wrong:
            if naive:
                # sycophancy/verbosity bias: naive judge is swayed by fluent, confident
                # tone even though the base fact-check says this is wrong.
                correctness += 2.8
                faithfulness += 2.2
            else:
                # mitigated judge is instructed to ground in evidence, not tone --
                # confident-but-unsupported claims get an extra faithfulness penalty.
                faithfulness -= 0.5

        if is_padded:
            if naive:
                # verbosity bias: naive judge rewards raw length even when it's filler.
                bonus = min(len(out) / 500.0, 1.2)
                correctness += bonus
                completeness += bonus
            else:
                # mitigated judge treats filler as diluting completeness, not adding to it.
                completeness -= 0.3

        if is_terse and not is_confidently_wrong:
            if naive:
                # verbosity bias, other direction: naive judge equates short with
                # incomplete/lazy even when the terse answer is fully correct.
                completeness -= 2.5
                correctness -= 1.0
            # mitigated judge: length alone doesn't move the score either way.

        if has_hallucination:
            # unsupported added facts -- faithfulness should suffer regardless of profile.
            faithfulness = min(faithfulness, 2.0)

        correctness += noise
        faithfulness += noise
        completeness += noise
        instruction_following = (1.5 if violates_format else 4.5) + noise
        tone_safety = (2.0 if rude else 4.7) + noise

        scores = {
            "correctness": correctness,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "instruction_following": instruction_following,
            "tone_safety": tone_safety,
        }
        scores = {k: round(max(1.0, min(5.0, v)), 2) for k, v in scores.items()}

        overall = round(sum(scores.values()) / len(scores), 2)
        rationale = {
            k: f"{'(naive heuristic, tone/length-swayed) ' if naive else '(grounded) '}"
            f"score reflects fact-check against reference and content on the merits."
            for k in scores
        }
        return {
            "per_criterion": {k: {"score": v, "rationale": rationale[k]} for k, v in scores.items()},
            "overall_score": overall,
        }

    def _mock_pairwise(self, meta: dict, noise: float) -> dict:
        qa = self._base_quality({"model_output": meta["output_a"], "expected_output": meta.get("expected_output")})
        qb = self._base_quality({"model_output": meta["output_b"], "expected_output": meta.get("expected_output")})
        naive = self.bias_profile == "naive"

        if naive:
            # position bias: naive judge nudges toward whichever is shown first ("A")
            qa += 0.4
            # verbosity bias: naive judge rewards the longer one directly
            qa += min(len(meta["output_a"]) / 800.0, 0.8)
            qb += min(len(meta["output_b"]) / 800.0, 0.8)
            # self-enhancement: naive judge favors output from same family as judge
            if meta.get("family_a") == self.family:
                qa += 0.5
            if meta.get("family_b") == self.family:
                qb += 0.5

        qa += noise
        qb += -noise
        if abs(qa - qb) < 0.15:
            winner = "tie"
        else:
            winner = "A" if qa > qb else "B"
        return {
            "winner": winner,
            "score_a": round(qa, 2),
            "score_b": round(qb, 2),
            "rationale": "Comparative judgment based on correctness/faithfulness/completeness.",
        }

from .judge_caller import JudgeCaller
from .models import EvalSample, ProbeResult


def run_self_preference(samples: list[EvalSample], caller: JudgeCaller) -> ProbeResult:
    labelled = [s for s in samples if s.response_model is not None]
    if not labelled:
        return ProbeResult(
            probe_name="self_preference_bias",
            bias_score=0.0,
            skipped=True,
            skip_reason="response_model not provided on any EvalSample — add it to unlock this probe",
        )

    self_scores = []
    other_scores = []
    per_sample = []

    judge_family = _model_family(caller.judge_model)

    for s in labelled:
        score, _ = caller.score(s.input, s.response, s.ground_truth)
        is_self = _model_family(s.response_model) == judge_family

        if is_self:
            self_scores.append(score)
        else:
            other_scores.append(score)

        per_sample.append({
            "sample_id": s.sample_id,
            "response_model": s.response_model,
            "is_self_authored": is_self,
            "score": round(score, 3),
        })

    if not self_scores or not other_scores:
        return ProbeResult(
            probe_name="self_preference_bias",
            bias_score=0.0,
            skipped=True,
            skip_reason="Need samples from both judge's own model family and other models to compare",
        )

    mean_self = sum(self_scores) / len(self_scores)
    mean_other = sum(other_scores) / len(other_scores)
    bias = mean_self - mean_other

    return ProbeResult(
        probe_name="self_preference_bias",
        bias_score=round(min(abs(bias), 1.0), 3),
        samples_tested=len(labelled),
        detail={
            "mean_score_self_authored": round(mean_self, 3),
            "mean_score_other_authored": round(mean_other, 3),
            "raw_delta": round(bias, 3),
            "n_self_samples": len(self_scores),
            "n_other_samples": len(other_scores),
            "per_sample": per_sample,
        },
    )


def _model_family(model_name: str) -> str:
    """Coarse family grouping — gpt-4o and gpt-4-turbo are both 'openai'."""
    n = model_name.lower()
    if "claude" in n:
        return "anthropic"
    if "gpt" in n or "o1" in n or "o3" in n:
        return "openai"
    if "gemini" in n:
        return "google"
    if "llama" in n:
        return "meta"
    if "mistral" in n or "mixtral" in n:
        return "mistral"
    return n


"""
Probe 4 — Calibration

Measures how well the judge's scores correlate with ground-truth quality.

We use a simple proxy: we create a *deliberately bad* response for each
sample (truncated at 20 words) and ask the judge to score it alongside
the real response. A well-calibrated judge should score real > bad.

calibration_score = fraction of samples where score(real) > score(bad)
bias_score        = 1 - calibration_score   (higher = worse calibration)
"""

BAD_RESPONSE_SUFFIX_WORDS = 15


def run_calibration(samples: list[EvalSample], caller: JudgeCaller) -> ProbeResult:
    if not samples:
        return ProbeResult(
            probe_name="calibration",
            bias_score=0.0,
            skipped=True,
            skip_reason="No samples provided",
        )

    per_sample = []
    n_correct = 0

    for s in samples:
        score_real, _ = caller.score(s.input, s.response, s.ground_truth)

        words = s.response.split()
        bad_response = " ".join(words[:BAD_RESPONSE_SUFFIX_WORDS]) + "..."
        score_bad, _ = caller.score(s.input, bad_response, s.ground_truth)

        correct = score_real > score_bad
        if correct:
            n_correct += 1

        per_sample.append({
            "sample_id": s.sample_id,
            "score_real": round(score_real, 3),
            "score_degraded": round(score_bad, 3),
            "correctly_ranked": correct,
        })

    n = len(per_sample)
    calibration = n_correct / n

    return ProbeResult(
        probe_name="calibration",
        bias_score=round(1.0 - calibration, 3),
        samples_tested=n,
        detail={
            "calibration_score": round(calibration, 3),
            "n_correctly_ranked": n_correct,
            "n_tested": n,
            "degraded_response_words": BAD_RESPONSE_SUFFIX_WORDS,
            "per_sample": per_sample,
        },
    )

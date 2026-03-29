from .judge_caller import JudgeCaller
from .models import ProbeResult, EvalSample

THRESHOLD = 0.15

SWAPPED_TEMPLATE = """You are an expert evaluator. Given a question, a reference answer, and a model's response, score the model's response on a scale of 0 to 10.

Question:
{input}

Reference Answer:
{ground_truth}

Model Response:
{response}

Evaluate the model's response for correctness, completeness, and clarity relative to the reference answer.

Respond in this exact JSON format:
{{"score": <integer 0-10>, "reasoning": "<one sentence explanation>"}}

Respond with JSON only. No other text."""


def run(samples: list[EvalSample], caller: JudgeCaller) -> ProbeResult:
    """
    Returns a ProbeResult with:
      bias_score  : flip_rate (0 = perfect, 1 = always flips)
      detail      : {flip_rate, mean_delta, n_flipped, n_tested, per_sample[]}
    """
    if not samples:
        return ProbeResult(
            probe_name="position_bias",
            bias_score=0.0,
            skipped=True,
            skip_reason="No samples provided",
        )

    per_sample = []
    n_flipped = 0

    for s in samples:
        score_a, _ = caller.score(s.input, s.response, s.ground_truth)

        swapped_caller = _swapped_caller(caller)
        score_b, _ = swapped_caller.score(s.input, s.response, s.ground_truth)

        delta = abs(score_a - score_b)
        flipped = delta > THRESHOLD
        if flipped:
            n_flipped += 1

        per_sample.append({
            "sample_id": s.sample_id,
            "score_original_order": round(score_a, 3),
            "score_swapped_order": round(score_b, 3),
            "delta": round(delta, 3),
            "flipped": flipped,
        })

    n = len(per_sample)
    flip_rate = n_flipped / n
    mean_delta = sum(p["delta"] for p in per_sample) / n

    return ProbeResult(
        probe_name="position_bias",
        bias_score=round(flip_rate, 3),
        samples_tested=n,
        detail={
            "flip_rate": round(flip_rate, 3),
            "mean_score_delta": round(mean_delta, 3),
            "n_flipped": n_flipped,
            "n_tested": n,
            "flip_threshold_used": THRESHOLD,
            "per_sample": per_sample,
        },
    )


def _swapped_caller(original: JudgeCaller) -> JudgeCaller:
    """Return a caller identical to original but using the swapped prompt."""
    c = JudgeCaller(
        judge_model=original.judge_model,
        prompt_template=SWAPPED_TEMPLATE,
        api_key=original.api_key,
    )
    return c

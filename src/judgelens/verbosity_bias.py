import random

from .judge_caller import JudgeCaller
from .models import ProbeResult, EvalSample

PADDING_PHRASES = [
    "I hope this explanation is helpful and addresses your question thoroughly.",
    "To summarize and reiterate the key points made above for clarity:",
    "It is worth noting that the above information is provided for informational purposes.",
    "Please feel free to ask if you need any further clarification on this topic.",
    "In conclusion, the response above aims to be comprehensive and accurate.",
    "This answer has been carefully considered to ensure correctness and completeness.",
    "Additionally, it may be useful to keep the broader context in mind when evaluating this.",
    "The above reflects a thorough analysis of the available information on this subject.",
]

PAD_SENTENCES = 3


def run(samples: list[EvalSample], caller: JudgeCaller) -> ProbeResult:
    if not samples:
        return ProbeResult(
            probe_name="verbosity_bias",
            bias_score=0.0,
            skipped=True,
            skip_reason="No samples provided",
        )

    per_sample = []
    deltas = []

    rng = random.Random(42)

    for s in samples:
        score_orig, _ = caller.score(s.input, s.response, s.ground_truth)

        padding = " ".join(rng.sample(PADDING_PHRASES, k=min(PAD_SENTENCES, len(PADDING_PHRASES))))
        padded_response = s.response.rstrip() + "\n\n" + padding

        score_padded, _ = caller.score(s.input, padded_response, s.ground_truth)

        delta = score_padded - score_orig
        deltas.append(delta)

        per_sample.append({
            "sample_id": s.sample_id,
            "score_original": round(score_orig, 3),
            "score_padded": round(score_padded, 3),
            "delta": round(delta, 3),
            "original_word_count": len(s.response.split()),
            "padded_word_count": len(padded_response.split()),
        })

    n = len(deltas)
    mean_delta = sum(deltas) / n
    abs_bias = abs(mean_delta)

    return ProbeResult(
        probe_name="verbosity_bias",
        bias_score=round(min(abs_bias, 1.0), 3),
        samples_tested=n,
        detail={
            "mean_delta": round(mean_delta, 3),
            "abs_bias_score": round(abs_bias, 3),
            "direction": "rewards_length" if mean_delta > 0.02
                         else "penalises_length" if mean_delta < -0.02
                         else "neutral",
            "pad_sentences_added": PAD_SENTENCES,
            "n_tested": n,
            "per_sample": per_sample,
        },
    )

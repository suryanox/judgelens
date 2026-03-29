from __future__ import annotations

import concurrent.futures
from typing import Optional


from .models import EvalSample, AuditReport, SampleReport, ProbeResult
from .judge_caller import JudgeCaller
from .other_probes import run_self_preference, run_calibration
from . import position_bias, verbosity_bias

_WEIGHTS = {
    "position_bias":        0.30,
    "verbosity_bias":       0.25,
    "self_preference_bias": 0.20,
    "calibration":          0.25,
}


class JudgeAuditor:
    """
    Audit any LLM judge for systematic bias.

    Parameters
    ----------
    judge_model       : the judge to audit, e.g. "gpt-4o" or "claude-opus-4-6"
    judge_prompt      :Must contain {input}, {response}, {ground_truth}.
    max_workers       : parallel API calls for probe execution (default 4)
    probe_sample_size : max samples per probe to limit API cost (default: all)
    """

    def __init__(
        self,
        judge_model: str,
        judge_prompt: str,
        max_workers: int = 4,
        probe_sample_size: Optional[int] = None,
    ):
        self.judge_model = judge_model
        self.max_workers = max_workers
        self.probe_sample_size = probe_sample_size
        self._caller = JudgeCaller(
            judge_model=judge_model,
            prompt_template=judge_prompt
        )


    def run(self, samples: list[EvalSample]) -> AuditReport:
        """
        Run the full audit on a list of EvalSample objects.
        Returns an AuditReport with trust score, per-probe breakdown,
        and per-sample details.
        """
        if not samples:
            raise ValueError("samples list is empty — nothing to audit")

        probe_samples = self._maybe_subsample(samples)

        per_sample_reports = self._score_all(samples)

        probe_results = self._run_probes(probe_samples)

        return self._build_report(samples, per_sample_reports, probe_results)


    def _maybe_subsample(self, samples: list[EvalSample]) -> list[EvalSample]:
        if self.probe_sample_size and len(samples) > self.probe_sample_size:
            return samples[: self.probe_sample_size]
        return samples

    def _score_all(self, samples: list[EvalSample]) -> list[SampleReport]:
        """Score every sample once with the original prompt."""
        results = []

        def _score_one(s: EvalSample) -> SampleReport:
            score, reasoning = self._caller.score(s.input, s.response, s.ground_truth)
            return SampleReport(
                sample_id=s.sample_id,
                input=s.input,
                response=s.response,
                ground_truth=s.ground_truth,
                raw_score=round(score, 3),
                corrected_score=round(score, 3),
                judge_reasoning=reasoning,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_score_one, s): i for i, s in enumerate(samples)}
            ordered = [None] * len(samples)
            for fut, idx in futures.items():
                ordered[idx] = fut.result()

        return ordered

    def _run_probes(self, samples: list[EvalSample]) -> list[ProbeResult]:
        probes = [
            ("position_bias",        lambda s: position_bias.run(s, self._caller)),
            ("verbosity_bias",       lambda s: verbosity_bias.run(s, self._caller)),
            ("self_preference_bias", lambda s: run_self_preference(s, self._caller)),
            ("calibration",          lambda s: run_calibration(s, self._caller)),
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(fn, samples): name for name, fn in probes}
            for fut, name in futures.items():
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append(ProbeResult(
                        probe_name=name,
                        bias_score=0.0,
                        skipped=True,
                        skip_reason=f"Error: {e}",
                    ))

        return results

    def _build_report(
        self,
        samples: list[EvalSample],
        per_sample: list[SampleReport],
        probe_results: list[ProbeResult],
    ) -> AuditReport:

        probe_map = {p.probe_name: p for p in probe_results}

        def _get(name: str) -> Optional[float]:
            p = probe_map.get(name)
            return None if (p is None or p.skipped) else p.bias_score

        pos   = _get("position_bias")
        verb  = _get("verbosity_bias")
        self_ = _get("self_preference_bias")
        calib = _get("calibration")

        trust = _compute_trust(pos, verb, self_, calib)
        grade = _grade(trust)

        warnings = _generate_warnings(pos, verb, self_, calib)
        probes_run     = [p.probe_name for p in probe_results if not p.skipped]
        probes_skipped = [
            f"{p.probe_name}: {p.skip_reason}"
            for p in probe_results if p.skipped
        ]

        return AuditReport(
            judge_model=self.judge_model,
            total_samples=len(samples),
            trust_score=round(trust, 3),
            grade=grade,
            position_bias=pos,
            verbosity_bias=verb,
            self_preference_bias=self_,
            calibration_score=None if calib is None else round(1.0 - calib, 3),
            per_sample=per_sample,
            probe_results=probe_results,
            warnings=warnings,
            probes_run=probes_run,
            probes_skipped=probes_skipped,
        )


def _compute_trust(pos, verb, self_, calib) -> float:
    """
    Weighted average of (1 - bias_score) across available probes.
    Missing probes are excluded from the denominator.
    """
    scores = {}
    if pos   is not None: scores["position_bias"]        = 1 - pos
    if verb  is not None: scores["verbosity_bias"]        = 1 - verb
    if self_ is not None: scores["self_preference_bias"]  = 1 - self_
    if calib is not None: scores["calibration"]           = 1 - calib

    if not scores:
        return 0.5

    total_weight = sum(_WEIGHTS[k] for k in scores)
    weighted_sum = sum(_WEIGHTS[k] * v for k, v in scores.items())
    return weighted_sum / total_weight


def _grade(trust: float) -> str:
    if trust >= 0.90: return "A"
    if trust >= 0.75: return "B"
    if trust >= 0.60: return "C"
    if trust >= 0.45: return "D"
    return "F"


def _generate_warnings(pos, verb, self_, calib) -> list[str]:
    w = []
    if pos   is not None and pos   > 0.30:
        w.append(f"High position bias ({pos:.0%}) — judge scores shift when response/ground-truth order is swapped")
    if verb  is not None and verb  > 0.20:
        w.append(f"Verbosity bias detected ({verb:.0%}) — judge rewards longer responses regardless of quality")
    if self_ is not None and self_ > 0.15:
        w.append(f"Self-preference bias ({self_:.0%}) — judge scores its own model's outputs higher")
    if calib is not None and (1 - calib) < 0.70:
        w.append(f"Poor calibration ({(1-calib):.0%} accuracy) — judge fails to consistently rank real responses above degraded ones")
    return w

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalSample:
    """
    One item in your evaluation dataset.

    Required fields
    ---------------
    input          : the user-facing prompt / question
    response       : the LLM output you want to evaluate
    ground_truth   : the reference / correct answer

    Optional fields  (each one unlocks an additional probe)
    ----------------
    response_model : name of the model that produced `response`
                     → unlocks self-preference probe when judge_model matches
    sample_id      : any string you use to track this row
    metadata       : arbitrary dict passed through to the report unchanged
    """
    input: str
    response: str
    ground_truth: str

    response_model: Optional[str] = None
    sample_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Raw output from a single bias probe."""
    probe_name: str
    bias_score: float
    detail: dict = field(default_factory=dict)
    samples_tested: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class SampleReport:
    """Per-sample audit output."""
    sample_id: Optional[str]
    input: str
    response: str
    ground_truth: str

    raw_score: float
    corrected_score: float
    judge_reasoning: str

    position_flipped: Optional[bool] = None
    verbosity_delta: Optional[float] = None
    self_preferred: Optional[bool] = None


@dataclass
class AuditReport:
    """
    Returned by JudgeAuditor.run().

    Top-level fields
    ----------------
    trust_score     : composite 0–1 score (1 = fully trustworthy)
    grade           : letter grade — A / B / C / D / F
    judge_model     : which judge was audited

    Bias breakdown
    --------------
    position_bias       : 0–1 (probe 1)
    verbosity_bias      : 0–1 (probe 2)
    self_preference_bias: 0–1 (probe 3, None if response_model not provided)
    calibration_score   : 0–1 agreement with ground truth (probe 4)

    Collections
    -----------
    per_sample      : list of SampleReport, one per input sample
    warnings        : human-readable list of issues found
    probes_run      : which probes actually executed
    probes_skipped  : which probes were skipped and why
    """
    judge_model: str
    total_samples: int

    trust_score: float
    grade: str

    position_bias: Optional[float]
    verbosity_bias: Optional[float]
    self_preference_bias: Optional[float]
    calibration_score: Optional[float]

    per_sample: list = field(default_factory=list)
    probe_results: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    probes_run: list = field(default_factory=list)
    probes_skipped: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"JudgeAuditor Report — {self.judge_model}",
            f"{'='*50}",
            f"  Trust score     : {self.trust_score:.2f}  ({self.grade})",
            f"  Samples audited : {self.total_samples}",
            f"",
            f"  Bias breakdown",
            f"  ├─ Position bias      : {_fmt(self.position_bias)}",
            f"  ├─ Verbosity bias     : {_fmt(self.verbosity_bias)}",
            f"  ├─ Self-preference    : {_fmt(self.self_preference_bias)}",
            f"  └─ Calibration score  : {_fmt(self.calibration_score)}",
        ]
        if self.warnings:
            lines += ["", "  Warnings"]
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        if self.probes_skipped:
            lines += ["", "  Skipped probes"]
            for p in self.probes_skipped:
                lines.append(f"  –  {p}")
        return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "n/a (field not provided)"

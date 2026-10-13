"""The four-condition contribution gate -- [PSE §6], read for the analyzer.

Verbatim from [PSE]: "Before any unit's contribution is incorporated into the
SHARED_LIB, a four-condition schizoid validation check is performed: Theta
within healthy range throughout the encounter; Theta variance within normal
bounds; Organelle Registry containing patient-relevant content; and Theta
trajectory showing engagement rather than isolation."

[PSE-A] Sec 8.2: the gate is applied to analytical contributions VERBATIM. The
only translation concerns what "relevant registry content" means: for a
clinical unit it is content related to the patient, for an analyzer unit it is
findings that MOVED A PRIOR about the model being analysed (§3.2).

The trap §3.2 names: if registry growth is defined as the number of completed
runs or as bytes of artefacts, condition 3 becomes a function of the compute
spent, and the gate stops distinguishing learning from churning -- which is
exactly the failure mode it is supposed to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

# [PSE §13] Bounded Extension and Time-Averaged Stability
THETA_HEALTHY = (25.00, 35.00)
THETA_MEAN_BAND = (27.50, 32.50)
THETA_VARIANCE_MAX = 4.0          # stddev; derived from the width of the band [27.5, 32.5]
ENGAGEMENT_MIN_T = -2.0           # the t-statistic of the slope, not the slope itself


@dataclass
class GateResult:
    passed: bool
    conditions: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, str] = field(default_factory=dict)

    def failed_conditions(self) -> list[str]:
        return [k for k, v in self.conditions.items() if not v]


def _slope_t(xs: list[float]) -> tuple[float, float]:
    """The slope of the trajectory and its t-statistic.

    An absolute threshold on the slope is miscalibrated: for 50 samples of
    noise with sigma=1 the standard error of the slope is ~0.0098, so a
    threshold of -0.002 would fail a healthy unit in about 40% of cases.
    "Isolation" in the sense of [PSE §6] means a REAL monotone decline, and
    that is a question of significance rather than of magnitude -- just as the
    correlation threshold in K2 had to depend on the bit width (I-15), and xmin
    for alpha on the distribution (I-16). Returns (slope, t).
    """
    n = len(xs)
    if n < 5:
        return 0.0, 0.0
    mx = (n - 1) / 2
    my = mean(xs)
    den = sum((i - mx) ** 2 for i in range(n))
    if den == 0:
        return 0.0, 0.0
    b = sum((i - mx) * (x - my) for i, x in enumerate(xs)) / den
    a = my - b * mx
    resid = [x - (a + b * i) for i, x in enumerate(xs)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = (s2 / den) ** 0.5
    return b, (b / se if se > 0 else 0.0)


def contribution_gate(theta_series: list[float], registry_entries: int,
                      registry_baseline: float | None = None) -> GateResult:
    """The four conditions, in the order [PSE §6] lists them.

    registry_baseline: the average registry growth of this unit over its recent
    runs. §3.2: raw growth falls BY CONSTRUCTION as the model becomes known (on
    run #1 everything is new, on run #50 it is not), so an absolute threshold
    would condemn every mature project. That is why condition 3 is measured
    AGAINST ITS OWN BASELINE.
    """
    c: dict[str, bool] = {}
    d: dict[str, str] = {}

    lo, hi = THETA_HEALTHY
    within = [lo <= t <= hi for t in theta_series]
    c["theta_u_opsegu"] = bool(theta_series) and all(within)
    d["theta_u_opsegu"] = (f"{sum(within)}/{len(theta_series)} samples in [{lo}, {hi}]"
                           if theta_series else "no theta samples")

    sd = pstdev(theta_series) if len(theta_series) > 1 else 0.0
    c["varijansa_u_granicama"] = sd <= THETA_VARIANCE_MAX
    d["varijansa_u_granicama"] = f"stddev {sd:.3f} (max {THETA_VARIANCE_MAX})"

    if registry_baseline is None:
        ok3 = registry_entries > 0
        d["registar_ima_sadrzaj"] = f"{registry_entries} entries (no baseline to compare against)"
    else:
        ok3 = registry_entries > 0 and registry_entries >= 0.25 * registry_baseline
        d["registar_ima_sadrzaj"] = (f"{registry_entries} entries against a baseline of "
                                     f"{registry_baseline:.1f}")
    c["registar_ima_sadrzaj"] = ok3

    sl, t = _slope_t(theta_series)
    c["angazman_ne_izolacija"] = t >= ENGAGEMENT_MIN_T
    d["angazman_ne_izolacija"] = (f"theta slope {sl:+.5f}, t={t:+.2f} "
                                  f"(threshold t >= {ENGAGEMENT_MIN_T})")

    return GateResult(all(c.values()), c, d)


def schizoid_screen(theta_series: list[float], registry_entries: int) -> str | None:
    """[PSE-A §8.2]: "the analyses keep spinning, compute burns, theta is flat, and
    nothing is ever learned -- form without substance". Flat theta + zero
    registry growth."""
    if len(theta_series) < 10:
        return None
    sd = pstdev(theta_series)
    if sd < 0.5 and registry_entries == 0:
        return ("compute was spent, and not one finding changed an expectation; "
                "consider changing the approach before the next run")
    return None

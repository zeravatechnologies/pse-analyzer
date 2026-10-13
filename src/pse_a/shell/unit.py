"""The analyzer unit -- [PSE §3], [PSE-A] Sec 8.

THE THETA CONVENTION (this matters): [PSE §13] says theta is in [25.00, 35.00],
nominally 30 -- that is the PERCENTAGE OF OTHER inside the boundary of the unit
(Self + Other = 100.00%). guardian/theta.py uses the inverted scale
(theta = 1 - other, nominally 0.70) because there "other" is the trust extended
to external entities. Here the [PSE] scale applies.

What "other" means for an analyzer unit ([PSE-A] 8.2: engagement = learning):
the model being analysed, and the findings about it that the unit has absorbed.
  - every finding that MOVES a prior (§3.2) adds weight to other (Empathic
    Infection)
  - without new findings the weight of other leaks away (Null State Drift,
    [PSE §13]): a unit that computes but does not learn sinks towards
    isolation, and that shows up in theta
This turns "compute burns, theta flat, nothing is ever learned" into a
measurable state rather than a description.

The GAIN and DECAY constants are calibrated (see calibrate()) so that a rate of
~0.1 findings per step holds theta around 30, while zero findings push the unit
out of range after ~45 steps. They are not derived from [PSE] -- [PSE] does not
give them -- but chosen and written down, so they can be changed through the
methodology-change procedure (>= 20 validated runs, [PSE-A] 8.4).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

OTHER_NOMINAL = 0.30
OTHER_CAP = 0.34            # below the upper bound of 35 -- Bounded Extension
DECAY = 0.004               # per step, Null State Drift
GAIN = 0.012                # per finding that moved a prior
NOISE = 0.0025              # No Immunity: the unit is always hit, so theta oscillates


@dataclass
class Budget:
    """The lifespan in compute steps ([PSE-A] 8.1). Exhaustion = death + harvest."""
    total: int
    spent: int = 0

    def step(self, n: int = 1) -> None:
        self.spent += n

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.total

    @property
    def remaining(self) -> int:
        return max(self.total - self.spent, 0)


@dataclass
class Unit:
    unit_id: str
    mission: dict
    budget: Budget
    other: float = OTHER_NOMINAL
    theta_series: list[float] = field(default_factory=list)
    registry: list[str] = field(default_factory=list)
    alive: bool = True
    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    @property
    def theta(self) -> float:
        return 100.0 * self.other

    def step(self, findings: int) -> float:
        """One compute step. findings = the number of findings that MOVED a prior."""
        self.other *= (1.0 - DECAY)
        self.other += GAIN * findings
        self.other += self._rng.gauss(0.0, NOISE)
        self.other = min(max(self.other, 0.0), OTHER_CAP)
        self.theta_series.append(self.theta)
        self.budget.step()
        return self.theta

    def absorb(self, entry_id: str) -> None:
        self.registry.append(entry_id)

    def die(self, reason: str) -> dict:
        """Mission Failure = Death. The unit is harvested; it does not blindly retry."""
        self.alive = False
        return {"unit": self.unit_id, "reason": reason,
                "steps": self.budget.spent, "registry": len(self.registry),
                "theta_last": self.theta_series[-1] if self.theta_series else None}


def calibrate(steps: int = 200, seeds: int = 50) -> dict:
    """Shows that GAIN/DECAY produce the required behaviour. Call after every change."""
    out = {}
    for rate in (0.0, 0.05, 0.1, 0.2, 0.4):
        finals, first_out = [], []
        for s in range(seeds):
            u = Unit(f"cal{s}", {}, Budget(steps), seed=s)
            rng = random.Random(1000 + s)
            fo = None
            for t in range(steps):
                u.step(1 if rng.random() < rate else 0)
                if fo is None and not (25.0 <= u.theta <= 35.0):
                    fo = t
            finals.append(u.theta)
            first_out.append(fo)
        outs = [x for x in first_out if x is not None]
        out[rate] = {"theta_final": sum(finals) / len(finals),
                     "leaves_range": len(outs) / seeds,
                     "first_exit_median": sorted(outs)[len(outs) // 2] if outs else None}
    return out

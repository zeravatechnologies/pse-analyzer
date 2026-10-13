"""Capability traffic light -- [SPEC] Sec 2, extended with time (I-2).

    ModelSource only                 -> L0, L1
    + Runtime with hooks             -> + L2, L3
    + Runtime with autograd          -> + L4

Invariant AZ-2: never fail in the middle of a run for a reason that was
knowable at L0. The extension: the estimated time IS such a reason, so it
goes into the traffic light.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cost import MachineRates, l1_sweep_seconds, human
from .source.canonical import ModelMap
from .source.gguf import GGUFFile


@dataclass
class Capabilities:
    layers: dict[str, bool] = field(default_factory=dict)
    estimates: dict[str, str] = field(default_factory=dict)
    blockers: dict[str, str] = field(default_factory=dict)

    def available(self) -> list[str]:
        return [k for k, v in self.layers.items() if v]


def semaphore(g: GGUFFile, mm: ModelMap, rates: MachineRates,
              ram_bytes: int, has_hooks: bool = False,
              has_autograd: bool = False, profile=None) -> Capabilities:
    c = Capabilities()

    # ---- L0 / L1: a ModelSource is enough
    c.layers["L0"] = True
    c.layers["L1"] = True
    shapes = [tuple(reversed(g.tensors[n].shape)) for n in mm.weight_matrices()]
    nev = profile.spike_eigenvectors if profile else 0
    est = l1_sweep_seconds(shapes, rates, g._file_size, eigenvectors=False)
    c.estimates["L1 sweep (values only)"] = f"{human(est['total_s'])}  ({est['n_matrices']} matrices)"
    if nev:
        est_v = l1_sweep_seconds(shapes, rates, g._file_size, eigenvectors=True)
        c.estimates["L1 sweep + spike vectors"] = human(est_v["total_s"])

    # ---- L2 / L3: need a Runtime with hooks, but before that it must FIT IN RAM.
    # This is why the 7B is L1-only (§1.3): not compute but memory.
    n_params = sum(g.tensors[n].n_elements for n in g.tensors)
    fp32_bytes = n_params * 4
    fits = fp32_bytes < ram_bytes * 0.55        # weights + activations + OS
    if not has_hooks:
        c.layers["L2"] = c.layers["L3"] = False
        c.blockers["L2"] = "no Runtime with hooks (torch is not available)"
    elif not fits:
        c.layers["L2"] = c.layers["L3"] = False
        c.blockers["L2"] = (f"the model in fp32 needs {fp32_bytes/1e9:.1f} GB, "
                            f"RAM is {ram_bytes/1e9:.1f} GB — it does not fit")
    else:
        c.layers["L2"] = c.layers["L3"] = True

    # ---- L4: needs autograd
    if not c.layers.get("L2"):
        c.layers["L4"] = False
        c.blockers["L4"] = c.blockers.get("L2", "L2 unavailable")
    elif not has_autograd:
        c.layers["L4"] = False
        c.blockers["L4"] = "the Runtime has no autograd"
    else:
        c.layers["L4"] = True
    return c

"""The L1 sweep as a PSE mission: unit, budget, priors, registry, seal.

[PSE-A] 8.1: every L1-L4 job is the mission of a single unit, with a lifespan
measured in compute steps. For L1 one step = one matrix.

The flow per matrix:
  dequantise (I-5) -> layer_health (core, clean) -> test the prior for every
  quantity (§3.2) -> a finding that moved an expectation goes into the registry
  -> a step of the unit carrying the number of such findings -> release the
  matrix

At the end: the artefact (a JSON report + the ESD histograms) -> the three
layers of sealing. If the gate refuses, the run stays WORKING STATE and does
not enter the dossier -- that is not a program error but a decision of the
system, and it is returned as such.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..core.source.gguf import open_gguf
from ..core.source.canonical import build_map
from ..core.source import dequant as dq
from ..core.spectral.esd import layer_health, ESD_BINS
from .shared_lib import SharedLib, SealRefused, Welford, config_hash
from .identity import model_identity
from .unit import Unit, Budget

QUANTITIES = ("alpha", "tail_frac", "log_sigma2", "log_stable_rank")


def _prior_key(family: str, role: str, depth: float | None, q: str) -> str:
    """The prior key = WHAT THE MODEL IS, not the precision it is written in.

    The ggml_type is a covariate in the report (I-8), but it is NOT part of the
    key: with the type in the key every rung of the ladder becomes its own silo,
    so re-measuring the same model in Q8_0 (correlation 0.99998 with f16) yields
    hundreds of "new" findings -- registry growth proportional to the number of
    rungs rather than to learning. The damage done by quantisation is measured
    by the separate qdiff mission, whose priors do include the type.

    family = architecture + width: the 0.5B and the 7B are both 'qwen2', but
    d_model 896 against 3584 changes both sigma² and alpha, so they must not
    share priors.
    """
    d = "g" if depth is None else f"d{round(depth * 4) / 4:.2f}"   # 5 depth bands
    return f"{family}/{role}/{d}/{q}"


def _values(h) -> dict[str, float]:
    return {
        "alpha": h.alpha,
        "tail_frac": h.tail_frac,
        "log_sigma2": math.log(h.sigma2) if h.sigma2 > 0 else float("nan"),
        "log_stable_rank": math.log(h.stable_rank) if h.stable_rank > 0 else float("nan"),
    }


def run_l1(model_path: str | Path, lib: SharedLib, max_matrices: int | None = None,
           include_embeddings: bool = False, seed: int = 0,
           progress=print) -> dict:
    model_path = Path(model_path)
    g = open_gguf(model_path)
    mm = build_map(g)
    mh = model_identity(lib, g)
    lib.register_model(mh, model_path.name, mm.architecture,
                       {n: vars(r) for n, r in mm.roles.items()})

    names = [n for n in mm.weight_matrices()
             if include_embeddings or mm.roles[n].role_class != "embedding"]
    names.sort(key=lambda n: (mm.roles[n].layer if mm.roles[n].layer is not None else -1, n))
    if max_matrices:
        names = names[:max_matrices]

    cfg = {"kind": "l1_sweep", "esd_bins": ESD_BINS, "n": len(names),
           "embeddings": include_embeddings, "quantities": QUANTITIES}
    run = lib.open_run(mh, "l1_sweep", cfg, {"seed": seed})
    unit = Unit(f"l1-{run.run_id}", {"state": "sweep", "model": mh,
                                     "action": "l1", "outcome": "report"},
                Budget(total=len(names)), seed=seed)

    family = f"{mm.architecture}-d{mm.d_model}"
    rows, hists = [], {}
    pending: list[tuple[str, float]] = []
    local: dict[str, Welford] = {}
    t0 = time.perf_counter()
    for i, nm in enumerate(names):
        r = mm.roles[nm]
        tname = g.tensors[nm].type_name
        W = dq.tensor(g, nm)
        h = layer_health(W, nm, tname, r.role, r.role_class, r.depth)
        del W

        moved = 0
        for q, v in _values(h).items():
            if not np.isfinite(v):
                continue
            key = _prior_key(family, r.role, r.depth, q)
            novel, z = lib.test_prior(key, float(v), local=local.get(key))
            pending.append((key, float(v)))
            local.setdefault(key, Welford()).push(float(v))
            if novel:
                lib.record_registry(run.run_id, f"layer_health/{q}", key, float(v),
                                    z if np.isfinite(z) else 1e9)
                unit.absorb(key)
                moved += 1
        unit.step(moved)
        lib.progress(run.run_id, i + 1, len(names), unit.theta, nm)

        rows.append({
            "name": nm, "role": r.role, "role_class": r.role_class, "layer": r.layer,
            "depth": r.depth, "ggml_type": tname, "shape": list(h.shape), "q": h.q,
            "sigma2": h.sigma2, "sigma_method": h.sigma_method, "edge": h.edge,
            "regime": h.regime, "n_tail": h.n_tail, "tail_frac": h.tail_frac,
            "alpha": h.alpha, "alpha_lo": h.alpha_lo, "alpha_hi": h.alpha_hi,
            "alpha_xmin": h.alpha_xmin, "stable_rank": h.stable_rank,
            "spectral_norm": h.spectral_norm, "moved_priors": moved,
            "mp_ks_n": h.mp_ks_n, "mp_fit_ok": h.mp_fit_ok,
            "theta": unit.theta,
        })
        hists[nm] = {"counts": h.esd_hist.tolist(), "edges": h.esd_edges.tolist()}
        if progress and (i % 10 == 0 or i == len(names) - 1):
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(names) - i - 1)
            progress(f"  [{i+1:3}/{len(names)}] {nm:28} {h.regime:13} "
                     f"theta={unit.theta:5.2f} registry={len(unit.registry):3} "
                     f"eta {eta/60:4.1f} min")

    elapsed = time.perf_counter() - t0
    report = {"model_hash": mh, "model": model_path.name, "architecture": mm.architecture,
              "n_layers": mm.n_layers, "d_model": mm.d_model, "d_ff": mm.d_ff,
              "kquant_clean": mm.kquant_clean, "type_inventory": g.type_inventory(),
              "config_hash": config_hash(cfg), "elapsed_s": elapsed, "matrices": rows}
    lib.attach(run, "l1_report.json", json.dumps(report, default=float).encode())
    lib.attach(run, "esd_histograms.json", json.dumps(hists).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())

    outcome = {"state": "sweep", "model": mh, "action": "l1", "outcome": "report",
               "n_matrices": len(rows)}
    spec = {"state": "sweep", "model": mh, "action": "l1", "outcome": "report",
            "n_matrices": len(names)}
    baseline = lib.registry_baseline(mh, "l1_sweep")
    result = {"run_id": run.run_id, "model_hash": mh, "elapsed_s": elapsed,
              "n_matrices": len(rows), "registry": len(unit.registry),
              "theta_min": min(unit.theta_series), "theta_max": max(unit.theta_series),
              "theta_mean": sum(unit.theta_series) / len(unit.theta_series)}
    try:
        gate = lib.seal(run, outcome, spec, unit.theta_series,
                        len(unit.registry), baseline, pending_priors=pending)
        result.update(sealed=True, gate=gate.detail)
    except SealRefused as e:
        result.update(sealed=False, refused=str(e), death=unit.die(str(e)))
        lib.record_death(run.run_id, str(e), len(unit.registry))
    from .physician import review
    result["prognoses"] = review(lib, run.model_hash if hasattr(run, "model_hash") else "", run.kind)
    return result

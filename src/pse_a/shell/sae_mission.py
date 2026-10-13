"""L3 as a PSE mission: harvest -> SAE -> the §9 profile -> feature cards -> seal.

[SPEC] Sec 6 + I-12. Acceptance of the dictionary lives in the PROFILE (the FVU
and the L0 range), because the [SPEC] constants are unreachable on small data
(§6 of the plan, 6.1). A dictionary that does not pass acceptance is NOT a
result: Mission Failure = Death, it is logged, and it is not blindly retried.

The corpus: a text file. For development on the Acer the default is
/usr/share/common-licenses (English legal prose, present on every Ubuntu) --
honestly named a development corpus. For the real science on the Dell a real
one is supplied.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from ..core.sae.sae import SAEConfig, train_sae, superposition_profile
from .harvest import HarvestStore, run_harvest
from .shared_lib import SharedLib, SealRefused, Welford
from .unit import Unit, Budget
from .lens_missions import _check_runtime


def dev_corpus() -> str:
    """English legal prose that already exists on the machine. See osenv for why
    this differs per platform; it stays honestly named a development corpus."""
    from .osenv import dev_corpus as _dev
    return _dev()


COACTIVATION_WINDOW = 5      # tokens ([SPEC] Sec 0)


def run_sae(model_path, lib: SharedLib, layer: int, profile, corpus_text: str | None = None,
            corpus_name: str = "dev: /usr/share/common-licenses", seed: int = 0,
            progress=print, supersedes: str | None = None) -> dict:
    from ..core.runtime.qwen2 import Qwen2Runtime
    from .doctor import data_root
    model_path = Path(model_path)
    g, mm, family = _check_runtime(model_path)
    mh = g.model_hash()
    lib.register_model(mh, model_path.name, mm.architecture, {})
    rt = Qwen2Runtime(model_path)
    text = corpus_text if corpus_text is not None else dev_corpus()
    ids = rt.encode(text)[: profile.harvest_tokens]
    chash = hashlib.sha256(np.asarray(ids, np.uint32).tobytes()).hexdigest()
    store = HarvestStore(data_root(), mh)
    if progress:
        progress(f"harvest: {len(ids)} tokens, layer {layer}, chunk {profile.chunk_tokens}, "
                 f"stride {profile.position_stride}")
    hv = run_harvest(rt, store, ids, chash, [layer], min(profile.chunk_tokens, 1024),
                     profile.position_stride, progress=None)

    cids = store.done_chunks(layer)
    base = store.baseline(layer)
    mu, sd = base.mean.astype(np.float32), np.maximum(base.std, 1e-6).astype(np.float32)
    X = np.concatenate([store.read(layer, c)[0].astype(np.float32) for c in cids])
    TOK = np.concatenate([store.read_tokens(c) for c in cids])
    Xn = (X - mu) / sd                                       # normalisation by Welford (SPEC 6)
    rng = np.random.default_rng(seed)
    sample = lambda n: Xn[rng.integers(0, Xn.shape[0], n)]

    cfg = SAEConfig(d=Xn.shape[1], expansion=profile.expansion_factor,
                    k=max(profile.l0_target[0], min(profile.l0_target[1], 32)),
                    k_aux=256, batch=profile.sae_batch, steps=profile.sae_lifespan,
                    lr=5e-4, dead_after=max(200, profile.sae_lifespan // 5),
                    resample_every=max(250, profile.sae_lifespan // 4), seed=seed)
    run = lib.open_run(mh, "l3_sae", {"kind": "l3_sae", "layer": layer, "profile": profile.name,
                                      "corpus": corpus_name, "corpus_hash": chash,
                                      "cfg": vars(cfg)}, {"seed": seed},
                           supersedes=supersedes)
    n_steps = max(1, cfg.steps // 250)
    unit = Unit(f"l3-{run.run_id}", {}, Budget(n_steps), seed=seed)
    pending, local = [], {}
    prev = {"fvu": None}

    def on_prog(step, met):
        # [PSE-A] 8.1: confirmation of progress (a plateau) SHORTENS the lifespan
        moved = 0
        key = f"{family}/sae/L{layer}/fvu_step"
        if prev["fvu"] is not None:
            gain = prev["fvu"] - met["fvu"]
            v = math.log1p(max(gain, 0.0) * 100)
            novel, z = lib.test_prior(key, v, local=local.get(key))
            local.setdefault(key, Welford()).push(v)
            if gain > 0.002:                                 # real progress = learning
                moved = 1
                unit.absorb(f"{key}@{step}")
        prev["fvu"] = met["fvu"]
        unit.step(moved)
        lib.progress(run.run_id, len(unit.theta_series), n_steps, unit.theta,
                     f"FVU {met['fvu']:.3f} L0 {met['l0']:.0f}")
        if progress and step % 1000 == 0:
            progress(f"  step {step}: FVU {met['fvu']:.4f}  L0 {met['l0']:.1f}  "
                     f"dead {100*met['dead_frac']:.1f}%")
        return "continue"

    res = train_sae(sample, cfg, sample(8192), on_progress=on_prog)
    accepted = (res.fvu <= profile.recon_fvu_max
                and profile.l0_target[0] <= res.l0 <= profile.l0_target[1])
    # The §9 profile measures co-activation within a window of CONSECUTIVE
    # tokens -- so over the ordering from the corpus, not over randomly shuffled
    # positions (the first version did that and reported "0 stable"; it was
    # replaced by a new run, AZ-5). The harvest takes every stride-th token, so
    # a window of 5 tokens fits into ~5/stride samples.
    win = max(1, round(COACTIVATION_WINDOW / max(profile.position_stride, 1)))
    seq = lambda n: Xn[: min(n, len(Xn))]
    prof = superposition_profile(res, seq, cfg, window=win,
                                 community_tag=f"{family}/L{layer}")
    prof["window_samples"] = win
    prof["window_tokens"] = win * profile.position_stride

    # the cards: where a feature fires most strongly, with its context
    import torch
    with torch.no_grad():
        Xt = torch.tensor(Xn[: min(len(Xn), 20000)])
        pre = (Xt - torch.tensor(res.b_dec)) @ torch.tensor(res.W_enc) + torch.tensor(res.b_enc)
        top = torch.topk(pre, cfg.k, dim=-1)
        Z = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values)).numpy()
    freq = (Z > 0).mean(0)
    order = np.argsort(-freq)
    cards = []
    for f in order[:64]:
        rows = np.argsort(-Z[:, f])[:6]
        ex = []
        for r in rows:
            if Z[r, f] <= 0:
                continue
            lo, hi = max(0, r - 6), min(len(TOK), r + 3)
            ex.append({"ctx": [rt.decode([int(t)]) for t in TOK[lo:hi]], "at": int(r - lo),
                       "act": float(Z[r, f])})
        cards.append({"feature": int(f), "freq": float(freq[f]),
                      "stable": int(f) in set(prof["stable_features"]), "examples": ex})

    buf = io.BytesIO()
    np.savez_compressed(buf, W_enc=res.W_enc, b_enc=res.b_enc, W_dec=res.W_dec,
                        b_dec=res.b_dec, mu=mu, sd=sd)
    lib.attach(run, "sae_dictionary.npz", buf.getvalue(), media="bin")
    report = {"model": model_path.name, "layer": layer, "profile": profile.name,
              "corpus": corpus_name, "n_tokens": int(len(TOK)), "harvest": hv,
              "fvu": res.fvu, "l0": res.l0, "dead_frac": res.dead_frac,
              "accepted": accepted, "fvu_max": profile.recon_fvu_max,
              "l0_target": list(profile.l0_target), "history": res.history,
              "superposition_profile": prof, "cards": cards, "d_model": int(Xn.shape[1]),
              "n_features": int(cfg.m)}
    lib.attach(run, "sae_report.json", json.dumps(report, default=float).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())

    if not accepted:
        why = (f"the dictionary does not pass acceptance: FVU {res.fvu:.3f} "
               f"(max {profile.recon_fvu_max}), L0 {res.l0:.1f} "
               f"(range {profile.l0_target})")
        lib.record_death(run.run_id, why, len(unit.registry))
        return {"sealed": False, "run_id": run.run_id, "refused": why,
                "death": unit.die(why), "fvu": res.fvu, "l0": res.l0}
    spec = {"action": "l3_sae", "accepted": True}
    out = {"action": "l3_sae", "accepted": accepted}
    try:
        lib.seal(run, out, spec, unit.theta_series, len(unit.registry),
                 lib.registry_baseline(mh, "l3_sae"), pending_priors=pending)
        return {"sealed": True, "run_id": run.run_id, "fvu": res.fvu, "l0": res.l0,
                "n_stable": prof["n_stable"], "n_eff": prof["n_eff"]}
    except SealRefused as e:
        lib.record_death(run.run_id, str(e), len(unit.registry))
        return {"sealed": False, "run_id": run.run_id, "refused": str(e),
                "death": unit.die(str(e)), "fvu": res.fvu, "l0": res.l0}

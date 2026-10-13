"""SAE -- the feature dictionary, [SPEC] Sec 6, with the I-12 corrections.

Why dead features blow up (§6 of the plan, phase 6.3): in a TopK SAE a feature
that never wins the top-k competition receives ZERO gradient and never
recovers -- death is an absorbing state. The measures, ordered by how much they
actually help:
  1. a smaller expansion on small data (a profile matter, not a code matter)
  2. AuxK: an auxiliary loss that forces the top-k_aux DEAD features to
     reconstruct the remaining error -> revival by gradient, not by re-init
  3. resampling done properly: the decoder onto an example with a LARGE loss,
     the encoder = the transposed decoder, and the Adam state for those
     parameters RESET
  4. the decoder initialised from the data, with unit column norms
Everything is a pure function (AZ-1): data in, dictionary and metrics out;
the budget arrives as a callback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SAEConfig:
    d: int
    expansion: int = 4
    k: int = 32
    k_aux: int = 256
    aux_coef: float = 1 / 32
    lr: float = 3e-4
    batch: int = 256
    steps: int = 5000
    dead_after: int = 2000          # steps without an activation -> dead
    resample_every: int = 1000
    seed: int = 0

    @property
    def m(self) -> int:
        return self.d * self.expansion


@dataclass
class SAEResult:
    W_enc: np.ndarray
    b_enc: np.ndarray
    W_dec: np.ndarray
    b_dec: np.ndarray
    fvu: float
    l0: float
    dead_frac: float
    history: list = field(default_factory=list)
    stopped: str = "budget"


def train_sae(sample_batch, cfg: SAEConfig, init_data: np.ndarray,
              on_progress=None) -> SAEResult:
    """sample_batch(n) -> np.ndarray (n, d), already normalised by the Welford statistics.

    on_progress(step, metrics) -> "continue" | "stop"  ([PSE-A] 8.1: a plateau
    SHORTENS the lifespan, divergence extends or splits it -- the decision is
    not made here).
    """
    import torch
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    d, m = cfg.d, cfg.m

    # decoder from the data, unit norms; encoder = the transposed decoder
    idx = rng.choice(init_data.shape[0], m, replace=init_data.shape[0] < m)
    Wd = torch.tensor(init_data[idx], dtype=torch.float32)
    Wd = Wd / Wd.norm(dim=1, keepdim=True).clamp_min(1e-8)          # (m, d)
    W_dec = torch.nn.Parameter(Wd.clone())
    W_enc = torch.nn.Parameter(Wd.T.clone())                         # (d, m)
    b_enc = torch.nn.Parameter(torch.zeros(m))
    b_dec = torch.nn.Parameter(torch.tensor(init_data.mean(0), dtype=torch.float32))
    params = [W_enc, b_enc, W_dec, b_dec]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    last_fired = np.zeros(m, dtype=np.int64)
    hist = []
    stopped = "budget"

    def encode(x):
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, cfg.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))
        return z, pre

    for step in range(1, cfg.steps + 1):
        x = torch.tensor(sample_batch(cfg.batch), dtype=torch.float32)
        z, pre = encode(x)
        xh = z @ W_dec + b_dec
        err = x - xh
        loss = (err ** 2).sum(-1).mean()

        fired = (z > 0).any(0).numpy()
        last_fired[fired] = step
        dead = (step - last_fired) > cfg.dead_after
        if dead.any() and cfg.aux_coef > 0:                 # AuxK
            dmask = torch.tensor(dead)
            pre_d = pre.masked_fill(~dmask, -1e9)
            ka = int(min(cfg.k_aux, dead.sum()))
            ta = torch.topk(pre_d, ka, dim=-1)
            za = torch.zeros_like(pre).scatter_(-1, ta.indices, torch.relu(ta.values))
            aux = ((err.detach() - za @ W_dec) ** 2).sum(-1).mean()
            loss = loss + cfg.aux_coef * aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():                               # unit decoder norms
            W_dec.div_(W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

        if step % cfg.resample_every == 0 and dead.any():
            _resample(W_enc, b_enc, W_dec, opt, dead, sample_batch, encode, b_dec, rng)
            last_fired[dead] = step

        if step % 250 == 0 or step == cfg.steps:
            with torch.no_grad():
                xv = torch.tensor(sample_batch(2048), dtype=torch.float32)
                zv, _ = encode(xv)
                ev = xv - (zv @ W_dec + b_dec)
                fvu = float((ev ** 2).sum() / ((xv - xv.mean(0)) ** 2).sum())
                l0 = float((zv > 0).sum(-1).float().mean())
            met = {"step": step, "fvu": fvu, "l0": l0,
                   "dead_frac": float(((step - last_fired) > cfg.dead_after).mean())}
            hist.append(met)
            if on_progress and on_progress(step, met) == "stop":
                stopped = "plateau"
                break

    with torch.no_grad():
        xv = torch.tensor(sample_batch(4096), dtype=torch.float32)
        zv, _ = encode(xv)
        ev = xv - (zv @ W_dec + b_dec)
        fvu = float((ev ** 2).sum() / ((xv - xv.mean(0)) ** 2).sum())
        l0 = float((zv > 0).sum(-1).float().mean())
    return SAEResult(W_enc.detach().numpy(), b_enc.detach().numpy(),
                     W_dec.detach().numpy(), b_dec.detach().numpy(),
                     fvu, l0, float(((cfg.steps - last_fired) > cfg.dead_after).mean()),
                     hist, stopped)


def _resample(W_enc, b_enc, W_dec, opt, dead, sample_batch, encode, b_dec, rng):
    """A dead feature -> the direction of a HIGH-LOSS EXAMPLE, plus an Adam state reset.

    A naive random re-init only kills them again: a new direction with no
    gradient of its own in the old Adam moment gets a step scaled by somebody
    else's history.
    """
    import torch
    with torch.no_grad():
        x = torch.tensor(sample_batch(8192), dtype=torch.float32)
        z, _ = encode(x)
        e = ((x - (z @ W_dec + b_dec)) ** 2).sum(-1)
        p = (e / e.sum()).numpy().astype(np.float64)
        p /= p.sum()
        di = np.nonzero(dead)[0]
        pick = rng.choice(x.shape[0], len(di), replace=len(di) > x.shape[0], p=p)
        v = x[pick] - b_dec
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        di_t = torch.tensor(di)
        W_dec[di_t] = v
        alive_norm = W_enc.norm(dim=0)[~torch.tensor(dead)].mean() if (~dead).any() else 1.0
        W_enc[:, di_t] = v.T * alive_norm * 0.2
        b_enc[di_t] = 0.0
        for prm in (W_enc, b_enc, W_dec):
            st = opt.state.get(prm)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if prm is W_enc:
                    st[key][:, di_t] = 0.0
                else:
                    st[key][di_t] = 0.0


def superposition_profile(res: SAEResult, sample_batch, cfg: SAEConfig,
                          window: int = 5, eps: float = 0.05, delta: float = 0.10,
                          community_tag: str = "", rel_thr: float = 0.5) -> dict:
    """[SPEC] Sec 6 export: the §9 junction. The SAE features become
    stable_features, the measured co-activation rates become the
    epsilon_profile, and the effective number of features becomes N_eff. From
    the first completed L3 run onwards [PSE §9] stops being a specification and
    becomes a measurement."""
    import torch
    x = torch.tensor(sample_batch(8192), dtype=torch.float32)
    W_enc = torch.tensor(res.W_enc); b_enc = torch.tensor(res.b_enc)
    b_dec = torch.tensor(res.b_dec)
    with torch.no_grad():
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, cfg.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))
    Z = z.numpy()
    # I-22: "active" in units of the feature's OWN typical magnitude. A TopK
    # with k above the true sparsity fills the surplus slots with tiny
    # activations that trail the main ones; those are real in the SAE's code but
    # not in the data, and they flood the profile (measured: k=4 at a sparsity
    # of 3.4 -> 1 out of 173 independent features counted as "stable"). The
    # per-feature threshold = rel_thr × the 90th percentile of its non-zero
    # activations.
    typ = np.full(Z.shape[1], np.inf)
    live = (Z > 0).any(0)
    if live.any():
        pos = np.where(Z[:, live] > 0, Z[:, live], np.nan)
        typ[live] = np.nanpercentile(pos, 90, axis=0)
    A = ((Z > 0) & (Z > rel_thr * typ[None, :])).astype(np.float32)
    freq = A.mean(0)
    alive = freq > 0
    # co-activation within a window of `window` consecutive samples
    n = (A.shape[0] // window) * window
    W = A[:n].reshape(-1, window, A.shape[1]).max(1)
    co = (W.T @ W) / max(W.shape[0], 1)
    fw = W.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(fw[:, None] > 0, co / fw[:, None], 0.0)   # P(j | i)
    # I-21: the EXCESS over independence, not the raw rate. In a window of
    # `window` samples with k active out of m, the CHANCE rate for any pair is
    # already ~k·window/m (for 4, 5, 256 that is 0.078 > eps), so an absolute
    # threshold would declare every feature superposed. The fifth time the same
    # pattern shows up (I-15, 16, 17, 20).
    cond = cond - fw[None, :]
    np.fill_diagonal(cond, 0.0)
    # ...and SIGNIFICANCE: the max over m-1 noisy estimates crosses eps by
    # chance alone (SE ~ sqrt(p(1-p)/n_i) ~ 0.024 for typical values). A pair is
    # superposed only if the excess is both meaningful (> eps) and significant
    # (z > the Bonferroni bound for m comparisons).
    n_i = W.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(np.maximum(fw * (1 - fw), 1e-12)[None, :] / np.maximum(n_i, 1)[:, None])
        z = np.where(se > 0, cond / se, 0.0)
    from scipy.stats import norm
    z_crit = float(norm.isf(0.05 / max(A.shape[1], 2)))
    # I-23: the normal approximation does not hold on small counts. With ~7
    # occurrences of a feature, ONE joint occurrence gives P(j|i)=0.14 and
    # z~3.7 -> "significant". Hence: a pair counts only with >= MIN_CO joint
    # occurrences, and a feature is evaluated at all only with >= MIN_N of its
    # own. The rest are "not enough data" -- neither stable nor superposed.
    MIN_N, MIN_CO = 20, 5
    counts = W.T @ W
    sig = (cond > eps) & (z > z_crit) & (counts >= MIN_CO)
    evaluable = alive & (n_i >= MIN_N)
    stable = [int(i) for i in np.nonzero(evaluable)[0] if not sig[i].any()]
    p = freq[alive] / freq[alive].sum()
    n_eff = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
    exceptional = float(sig[alive][:, alive].mean()) if alive.any() else 0.0
    return {"n_eff": n_eff, "n_alive": int(alive.sum()), "n_total": int(A.shape[1]),
            "stable_features": stable, "n_stable": len(stable), "z_crit": z_crit,
            "n_evaluable": int(evaluable.sum()),
            "epsilon": eps, "delta": delta, "exceptional_share": exceptional,
            "delta_stable": exceptional < delta, "community_tag": community_tag,
            "coactivation_top": _top_pairs(cond, 50)}


def _top_pairs(cond: np.ndarray, n: int) -> list:
    iu = np.dstack(np.unravel_index(np.argsort(-cond.ravel())[:n], cond.shape))[0]
    return [(int(i), int(j), float(cond[i, j])) for i, j in iu if cond[i, j] > 0]

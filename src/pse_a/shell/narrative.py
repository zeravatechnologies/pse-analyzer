"""The human register -- AZ-7, [SPEC] 11.2.

"Every result tells two stories": the mathematical one (the panels) and the
human one (here). Both come from the SAME artefacts; every claim carries the
hash of the artefact it comes from, so the two registers cannot drift apart.
The generator is DETERMINISTIC: the same sealed artefacts -> the same text, so
the story is as immutable as the seal, and it is computed without a language
model (no LLM call when a page is opened, §7.4).

The rules of [PSE §5] hold: domain language, the uncertainty inside the
sentence itself, no names of forms and no diagnoses (the AZ-8 test runs through
this module too), and a bounded length per section.
"""
from __future__ import annotations

import math


def _c(h: str) -> str:
    return f"[{h[:8]}]"


def _pct(x: float, nd: int = 1) -> str:
    return f"{100*x:.{nd}f} %"


def l1(rep: dict, h: str) -> list[str]:
    M = rep["matrices"]
    reg = {}
    for m in M:
        reg[m["regime"]] = reg.get(m["regime"], 0) + 1
    ht = reg.get("heavy_tailed", 0) + reg.get("mixed", 0)
    al = [m["alpha"] for m in M if m.get("alpha") is not None and math.isfinite(m["alpha"])]
    by_role = {}
    for m in M:
        if m.get("alpha") is not None and math.isfinite(m["alpha"]):
            by_role.setdefault(m["role"], []).append(m["alpha"])
    roles = sorted(((sum(v) / len(v), r) for r, v in by_role.items()))
    mp_ok = sum(1 for m in M if m.get("mp_fit_ok"))
    out = [f"{len(M)} weight matrices of the model {rep['model']} were examined {_c(h)}."]
    out.append(f"In {ht} of them the spectrum has a heavy tail, which means that training wrote "
               f"structure into them; the rest are closer to random {_c(h)}.")
    if roles:
        out.append(f"The most pronounced structure, by average alpha, is carried by {roles[0][1]} "
                   f"(≈{roles[0][0]:.1f}), and the weakest by {roles[-1][1]} (≈{roles[-1][0]:.1f}); "
                   f"the smaller alpha is, the more pronounced the structure {_c(h)}.")
    if "mp_fit_ok" in M[0]:
        out.append(f"In {mp_ok} of {len(M)} matrices the body of the spectrum really does look like "
                   f"noise of one single strength; for the others that picture does not hold, so "
                   f"for them we rely on alpha alone {_c(h)}.")
    if not rep.get("kquant_clean", True):
        out.append("Note: the dimensions of this model are not divisible by 256, so its quantised "
                   "versions differ in composition, not only in precision.")
    return out


def qdiff(rep: dict, h: str) -> list[str]:
    S = rep["strata"]
    worst = min(S.items(), key=lambda kv: kv[1]["kept_quarter"])
    best = max(S.items(), key=lambda kv: kv[1]["kept_quarter"])
    conc = [k.split("|")[0] for k, v in S.items() if v["noise_ratio"] > 1.2]
    dE = sum(v["energy_change"] * v["n"] for v in S.values()) / max(sum(v["n"] for v in S.values()), 1)
    out = [f"Two precisions of the same model were compared: {rep['ref']} and {rep['test']} {_c(h)}."]
    lab = lambda k: "{} ({})".format(k.split("|")[0], k.split("|")[1].replace("->", "→"))
    out.append(f"The principal directions of the matrices are largely preserved: most was kept by "
               f"{lab(best[0])}, {best[1]['kept_quarter']:.3f}, and least by "
               f"{lab(worst[0])}, {worst[1]['kept_quarter']:.3f} {_c(h)}.")
    out.append(f"The total energy of the weights changes on average by {_pct(dE, 2)} — quantisation "
               f"does not move it around, it reduces it slightly {_c(h)}.")
    if conc:
        out.append(f"For {', '.join(sorted(set(conc)))} the quantisation error falls more often on "
                   f"the learned directions themselves than on the rest {_c(h)}.")
    out.append("Whether this is felt in the behaviour of the model, the spectrum alone does not say; "
               "for that the output of the model on text has to be measured as well.")
    return out


def lens(rep: dict, h: str) -> list[str]:
    it = rep["items"]
    out = [f"For {len(it)} short sentences we looked at what the model would say if it had to speak "
           f"at every layer {_c(h)}."]
    for x in it[:3]:
        out.append(f"“{x['prompt']}” — the final choice “{x['top'][-1][-1][0]}” appears for the "
                   f"first time at layer {x['commit_layer']} {_c(h)}.")
    out.append("This view sees only what is already speakable; computation that is not yet in words "
               "stays hidden from it.")
    return out


def jspace(m: dict, h: str) -> list[str]:
    pl = m["per_layer"]
    vs = [pl[str(L)]["var_frac"] for L in m["layers"]]
    out = [f"We measured how much of its internal state the model holds in a form ready to be "
           f"spoken, over {len(m['layers'])} layers and {len(m['prompts'])} deliberately ambiguous "
           f"sentences {_c(h)}."]
    out.append(f"That share lies between {_pct(min(vs))} and {_pct(max(vs))} {_c(h)}.")
    r = m["regions"]
    if r.get("ignition") is not None:
        out.append(f"At layer {r['ignition']} the model commits sharply to one reading {_c(h)}.")
    elif r.get("resolution_ok") is False:
        out.append(f"No sharp moment of commitment was observed, but the map is coarse (every "
                   f"{r.get('stride')}th layer), so this is not proof that there is none {_c(h)}.")
    else:
        out.append(f"No sharp moment of commitment was observed; for this model the division into "
                   f"a sensory part, a workspace and an output has not been confirmed {_c(h)}.")
    return out


def sae(rep: dict, h: str) -> list[str]:
    p = rep["superposition_profile"]
    out = [f"From {rep['n_tokens']} positions at layer {rep['layer']} a dictionary of "
           f"{rep['n_features']} features was learned {_c(h)}."]
    out.append(f"The dictionary reconstructs the state with a remaining error of {_pct(rep['fvu'])}, "
               f"with about {rep['l0']:.0f} active features per position {_c(h)}.")
    if "n_evaluable" in p:
        out.append(f"For {p['n_evaluable']} of the {p['n_alive']} live features there are enough "
                   f"occurrences to judge; of those, {p['n_stable']} do not appear together with "
                   f"others more often than chance would give {_c(h)}.")
    else:
        out.append(f"Of the {p['n_alive']} live features, {p['n_stable']} do not appear together "
                   f"with others more often than chance would give {_c(h)}.")
    if not rep.get("accepted", True):
        out.append("The dictionary did not pass acceptance, so it did not enter knowledge.")
    return out


def ppl(rep: dict, h: str) -> list[str]:
    R = rep["rows"]
    out = [f"The same text of {rep['n_tokens']} tokens was given to each of the {len(R)} versions "
           f"of the model {_c(h)}."]
    worst = max(R[1:], key=lambda r: r["ppl_rel"]) if len(R) > 1 else None
    if worst:
        out.append(f"The one that suffers most is {worst['file']}: perplexity rises by "
                   f"{_pct(worst['ppl_rel'])} relative to the reference {_c(h)}.")
    both = [r for r in R[1:] if "kept_quarter_mean" in r]
    if len(both) >= 2:
        both.sort(key=lambda r: 1 - r["kept_quarter_mean"])
        same = [r["ppl_rel"] for r in both] == sorted(r["ppl_rel"] for r in both)
        out.append("The ordering by loss of the principal directions "
                   + ("matches" if same else "does not match")
                   + f" the ordering by degradation of the output, over {len(both)} compared "
                     f"versions {_c(h)}.")
    if not rep.get("kquant_clean", True):
        out.append("The versions of this model differ in composition, not only in precision, so "
                   "this is a comparison of files and not a clean curve over bit width.")
    return out


GEN = {"l1_sweep": ("l1_report.json", l1), "qdiff": ("qdiff_report.json", qdiff),
       "l2_lens": ("lens_report.json", lens), "l4_jspace": ("jspace_map.json", jspace),
       "l3_sae": ("sae_report.json", sae), "ppl_ladder": ("ppl_ladder.json", ppl)}


def story(kind: str, artifacts: dict, load) -> list[str]:
    if kind not in GEN:
        return []
    name, fn = GEN[kind]
    h = artifacts.get(name)
    if not h:
        return []
    return fn(load(h), h)

"""The only emitter of text towards the human -- AZ-8 / [PSE-A] Sec 9.

"No game form is ever named to the human -- not in text, and not in the GUI."
The form layer returns (form_id, posterior, horizon); here that is translated
into PROGNOSTIC DOMAIN LANGUAGE. No other module may build sentences about the
state of a unit. tests/test_az8.py greps everything this module can emit, all
the templates and all the SSE schemas against the forbidden vocabulary.
"""
from __future__ import annotations


def _runs(h: float | None) -> str:
    if h is None:
        return ""
    n = max(1, round(h))
    return {1: "roughly one run", 2: "roughly two runs",
            3: "roughly three runs"}.get(n, f"roughly {n} runs")


def prognosis(form_id: str, posterior: float, horizon: float | None = None) -> str:
    """(form, posterior, horizon) -> a sentence. The uncertainty stays in the sentence."""
    sure = "probably" if posterior >= 0.7 else "possibly"
    if form_id == "war_of_attrition":
        return (f"This line of work {sure} does not converge; {_runs(horizon)} to a "
                "decision point — change the approach or stop.")
    if form_id == "signalling":
        return (f"The results {sure} look better than they are verifiable; "
                "it is worth repeating them before they enter knowledge.")
    if form_id == "stag_hunt":
        return ("Confidence in the flow of the analysis has been shaken; a review of the "
                "log and a repeat on demand are available for every result.")
    if form_id == "iterated_pd":
        return "A unit close to the end of its budget finishes early, so as not to cut corners."
    if form_id == "ess":
        return "The default methodology is not changed until there are enough confirmed runs."
    return "The work is proceeding at its usual rhythm."


_GATE_TEXT = {
    "theta_u_opsegu": "the engagement of the unit was leaving the healthy range",
    "varijansa_u_granicama": "the engagement of the unit was too unstable",
    "registar_ima_sadrzaj": "the run did not change a single expectation about the model",
    "angazman_ne_izolacija": "the engagement of the unit declined while it worked",
}


def refusal(failed: list[str]) -> str:
    """Why a result was not accepted into knowledge -- with no clinical or strategic names."""
    parts = [_GATE_TEXT.get(f, "a check did not pass") for f in failed]
    return "The result was not accepted into knowledge: " + "; ".join(parts) + "."


def regime_label(regime: str) -> str:
    return {"random_like": "random-like", "spiked": "spike directions",
            "heavy_tailed": "heavy tail", "mixed": "mixed"}.get(regime, regime)


RUN_KIND = {"l1_sweep": "spectral map", "qdiff": "precision comparison",
            "l2_lens": "thought across layers", "l3_sae": "feature dictionary",
            "l4_jspace": "map of J-space"}


def run_kind_label(kind: str) -> str:
    return RUN_KIND.get(kind, kind)


def all_phrases() -> list[str]:
    """Everything the module can emit -- for the AZ-8 test."""
    out = [prognosis(f, p, h) for f in ("war_of_attrition", "signalling", "stag_hunt",
                                         "iterated_pd", "ess", "nash_substrate")
           for p in (0.5, 0.9) for h in (None, 1, 2, 5)]
    out += [refusal([k]) for k in _GATE_TEXT]
    out += [regime_label(r) for r in ("random_like", "spiked", "heavy_tailed", "mixed")]
    return out

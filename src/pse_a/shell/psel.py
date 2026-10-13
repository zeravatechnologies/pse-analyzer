"""PSE-L -- the voice of the system towards the human ([SPEC] 10.1, [PSE §5]).

"The conversation is the spine; panels are what the LLM summons." The model is
given ONLY summaries of sealed results (the stories from narrative.py and a few
numbers), never weights, activations or working state. It can open a panel with
the marker [[panel:/run/<id>/<view>]], which the GUI turns into a button
(panel.open).

AZ-8 must not depend on whether the model obeyed its instructions: every answer
goes through the SAME scan of the forbidden vocabulary as the templates and the
SSE, and a forbidden expression is stripped before the human ever sees it. That
is the only guarantee that holds for any model, local or somebody else's.
"""
from __future__ import annotations

import re
import time
import uuid

from . import narrative
from .prognostics import run_kind_label

PANEL = re.compile(r"\[\[panel:(/(?:run|model)/[A-Za-z0-9_./-]+|/inbox)\]\]")
VIEW = {"l1_sweep": "spectral", "qdiff": "qdiff", "l2_lens": "lens", "l3_sae": "sae",
        "l4_jspace": "jspace", "ppl_ladder": "ppl"}

SYSTEM = """You are PSE-L, the voice of PSE-A, an analytical system that studies local \
language models (the spectrum of the weight matrices, comparison of precisions, thought across \
layers, the feature dictionary, J-space).

Rules:
- Answer in the user's language (English by default), briefly and clearly.
- Rely EXCLUSIVELY on the results listed below. If something was not measured, say so.
- The uncertainty goes into the sentence itself ("probably", "in this sample", "at this \
resolution").
- When you base a claim on a result, cite the artefact tag in square brackets, e.g. [59286d39].
- To open a panel, write the marker exactly like this: [[panel:/run/<run_id>/<view>]] (the paths \
are below).
- Do not invent numbers, models or findings.
- Do not use the names of theoretical frameworks or clinical terms; speak about the flow of the \
work, convergence, learning and the repeatability of results.

SEALED RESULTS:
{context}
"""


def budget_for(pid: str) -> int:
    """How many characters of context may go into the system message. Local servers
    (Ollama defaults to 4096 tokens) SILENTLY cut the excess off the FRONT -- and
    that is where the rules are. So the context is short for local models and
    wider for online ones."""
    from .llm import PROVIDERS
    # "lokalno" is the group lookup key in PROVIDERS, not display text.
    grp = PROVIDERS.get(pid, ("", "online"))[1]
    return 5000 if grp == "lokalno" else 24000


def build_context(lib, max_runs: int = 12, max_chars: int = 24000) -> str:
    rows = lib.db.execute(
        "SELECT r.run_id, r.kind, m.display_name FROM runs r JOIN models m "
        "ON m.model_hash = r.model_hash WHERE r.state='sealed' "
        "ORDER BY r.sealed DESC LIMIT ?", (max_runs,)).fetchall()
    import json as _j
    load = lambda h: _j.loads(lib.get_blob(h))
    parts = []
    for run_id, kind, name in rows:
        arts = dict(lib.db.execute("SELECT name, content_hash FROM artifacts WHERE run_id=?",
                                   (run_id,)).fetchall())
        story = narrative.story(kind, arts, load)
        path = f"/run/{run_id}/{VIEW.get(kind, '')}"
        parts.append(f"- {run_kind_label(kind)} · {name} · panel {path}\n  " + "\n  ".join(story))
    inbox = lib.inbox(unseen_only=True)
    if inbox:
        parts.append("UNREAD MAIL:\n" + "\n".join(f"- {i['text']}" for i in inbox[:5]))
    out, used = [], 0
    for part in parts:                         # newest first; the old ones drop out whole
        if used + len(part) > max_chars:
            out.append(f"({len(parts) - len(out)} older results omitted because of size)")
            break
        out.append(part)
        used += len(part)
    return "\n".join(out) if out else "(there are no sealed results yet)"


def enforce_az8(text: str) -> tuple[str, int]:
    """The mechanical AZ-8 guarantee for the output of ANY model."""
    from ..tests.test_az8 import FORBIDDEN
    n = 0
    for pat in FORBIDDEN:
        text, k = re.subn(pat, "[removed]", text, flags=re.IGNORECASE)
        n += k
    return text, n


SCHEMA = """CREATE TABLE IF NOT EXISTS chat (
    msg_id TEXT PRIMARY KEY, role TEXT, text TEXT, provider TEXT, model TEXT, t REAL)"""


def history(lib, n: int = 40) -> list[dict]:
    lib.db.execute(SCHEMA)
    rows = lib.db.execute("SELECT role, text, provider, model, t FROM chat "
                          "ORDER BY t DESC LIMIT ?", (n,)).fetchall()[::-1]
    return [dict(zip(("role", "text", "provider", "model", "t"), r)) for r in rows]


def append(lib, role: str, text: str, provider: str = "", model: str = "") -> None:
    """The conversation is append-only, like the notes; nothing is rewritten."""
    lib.db.execute(SCHEMA)
    lib.db.execute("INSERT INTO chat VALUES (?,?,?,?,?,?)",
                   (uuid.uuid4().hex[:16], role, text, provider, model, time.time()))
    lib.db.commit()


def respond(lib, pid: str, entry: dict, user_text: str):
    """Generator: pieces of the answer, already filtered. At the end it stores the whole answer."""
    from . import llm
    system = SYSTEM.format(context=build_context(lib))
    past = [{"role": m["role"], "content": m["text"]} for m in history(lib, 12)
            if m["role"] in ("user", "assistant")]
    msgs = past + [{"role": "user", "content": user_text}]
    append(lib, "user", user_text)
    full = ""
    try:
        for piece in llm.chat(pid, entry, system, msgs):
            full += piece
        clean, n = enforce_az8(full)
        if n:
            clean += ("\n\n(Part of the answer was removed: it contained expressions this "
                      "system does not display.)")
    except llm.LLMError as e:
        clean = f"I could not get an answer: {e}"
    append(lib, "assistant", clean, pid, entry.get("model") or entry.get("gguf", ""))
    return clean


def render(text: str) -> list[tuple[str, str]]:
    """Text -> a sequence of (kind, content): plain text and buttons for panels."""
    out, pos = [], 0
    for m in PANEL.finditer(text):
        if m.start() > pos:
            out.append(("text", text[pos:m.start()]))
        out.append(("panel", m.group(1)))
        pos = m.end()
    if pos < len(text):
        out.append(("text", text[pos:]))
    return out


def respond_stream(lib, pid: str, entry: dict, user_text: str):
    """Like respond(), but yields ("partial", text) as it arrives, then ("done", text).

    AZ-8 holds for PARTIAL text too: only filtered text up to the last COMPLETED
    word is sent, so a forbidden expression never shows up even for a moment --
    not even while the answer is still being typed. Multi-word expressions are
    caught as soon as they complete, because the filter is always applied to the
    whole text so far.
    """
    from . import llm
    system = SYSTEM.format(context=build_context(lib, max_chars=budget_for(pid)))
    n_hist = 4 if budget_for(pid) < 10000 else 12
    past = [{"role": m["role"], "content": m["text"]} for m in history(lib, n_hist)
            if m["role"] in ("user", "assistant")]
    append(lib, "user", user_text)
    full, clean = "", ""
    try:
        for piece in llm.chat(pid, entry, system, past + [{"role": "user", "content": user_text}]):
            full += piece
            cut = max(full.rfind(" "), full.rfind("\n"))
            if cut > 0:
                shown, _ = enforce_az8(full[:cut])
                yield "partial", shown
        clean, n = enforce_az8(full)
        if n:
            clean += ("\n\n(Part of the answer was removed: it contained expressions this "
                      "system does not display.)")
    except llm.LLMError as e:
        clean = (full and enforce_az8(full)[0] + "\n\n") + f"I could not finish the answer: {e}"
    append(lib, "assistant", clean, pid, entry.get("model") or entry.get("gguf", ""))
    yield "done", clean

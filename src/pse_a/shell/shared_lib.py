"""SHARED_LIB -- the orchestrating intelligence ([PSE §6], [PSE-A] Sec 8.4).

This is NOT the Hub class from guardian/shared_lib.py. The Hub is a system for
signing and revoking capabilities; this is a store of knowledge, artefacts and
dossiers.

Invariants that are BUILT IN, not checked after the fact:
  AZ-5  what is sealed is immutable -- no method exists that changes an artefact
  AZ-6  notes are append-only, attributed, timestamped and anchored to a hash
  AZ-4  an artefact without (model_hash, config_hash, seeds) is not a result
  I-9   Welford is DERIVED from the per-chunk triples, not updated in place

There is no update_artifact, no delete_run and no edit_note. A correction is a
NEW run with supersedes=<run_id>; both stay visible and the dossier shows the
lineage.
"""
from __future__ import annotations
from ..core.source.gguf import MODEL_HASH_VERSION

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contribution import GateResult, contribution_gate

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_hash TEXT PRIMARY KEY, display_name TEXT, architecture TEXT,
    canonical_map TEXT, created REAL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, model_hash TEXT, kind TEXT, config_hash TEXT,
    seeds TEXT, started REAL, sealed REAL, supersedes TEXT,
    budget_spent REAL, state TEXT);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT, name TEXT, content_hash TEXT, n_bytes INTEGER, media TEXT,
    PRIMARY KEY (run_id, name));
CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY, run_id TEXT, author TEXT, anchor_hash TEXT,
    text TEXT, created REAL);
CREATE TABLE IF NOT EXISTS priors (
    key TEXT PRIMARY KEY, n INTEGER, mean REAL, m2 REAL, updated REAL);
CREATE TABLE IF NOT EXISTS registry (
    entry_id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, prior_key TEXT,
    value REAL, z REAL, created REAL);
CREATE TABLE IF NOT EXISTS progress (
    run_id TEXT, step INTEGER, total INTEGER, theta REAL, label TEXT, t REAL,
    PRIMARY KEY (run_id, step));
CREATE TABLE IF NOT EXISTS deaths (
    run_id TEXT PRIMARY KEY, reason TEXT, registry INTEGER, t REAL);
CREATE TABLE IF NOT EXISTS inbox (
    item_id TEXT PRIMARY KEY, kind TEXT, model_hash TEXT, run_kind TEXT,
    text TEXT, t REAL, seen INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS hash_migrations (
    old_hash TEXT PRIMARY KEY, new_hash TEXT, t REAL);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_hash);
CREATE INDEX IF NOT EXISTS idx_reg_run ON registry(run_id);
"""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def config_hash(cfg: dict) -> str:
    return sha256_bytes(json.dumps(cfg, sort_keys=True, default=str).encode())


@dataclass
class Welford:
    """I-9: the triple (n, mean, M2) that merges PAIRWISE (Chan).

    This is why the baseline can be derived rather than primary: every chunk
    carries its own triple, the global baseline is their merge over the done
    set from the manifest, so it is always exactly consistent with what is on
    disk.
    """
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float) -> "Welford":
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        return self

    @staticmethod
    def merge(a: "Welford", b: "Welford") -> "Welford":
        if a.n == 0:
            return Welford(b.n, b.mean, b.m2)
        if b.n == 0:
            return Welford(a.n, a.mean, a.m2)
        n = a.n + b.n
        d = b.mean - a.mean
        return Welford(n, a.mean + d * b.n / n, a.m2 + b.m2 + d * d * a.n * b.n / n)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


class SealRefused(Exception):
    """The three layers of verification did not pass ([PSE-A] 11.4)."""


@dataclass
class RunRecord:
    run_id: str
    model_hash: str
    kind: str
    config_hash: str
    seeds: dict
    started: float
    state: str = "working"            # working -> sealed; never back
    sealed: float | None = None
    supersedes: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)   # name -> content_hash


class SharedLib:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        self._path = self.root / "shared_lib.sqlite"
        self._tls = threading.local()
        self.db.executescript(SCHEMA)
        self._upgrade_schema()
        self.db.commit()
        self._flag_legacy_models()

    @property
    def db(self) -> sqlite3.Connection:
        """One connection PER THREAD. The web layer runs its endpoints in a
        threadpool, and a sqlite connection must not cross a thread boundary.
        WAL allows several concurrent readers -- even while a mission in another
        process is writing."""
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            self._tls.conn = c
        return c

    def close(self) -> None:
        """Release this thread's connection.

        Needed on Windows, where an open handle makes the file undeletable: a
        temporary directory cannot be cleaned up while sqlite still holds it.
        """
        c = getattr(self._tls, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
            self._tls.conn = None

    def __enter__(self) -> "SharedLib":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------- artefacts
    def _blob_path(self, h: str) -> Path:
        d = self.root / "artifacts" / h[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / h

    def put_blob(self, data: bytes) -> str:
        """A content-addressed write. The same content -> the same hash -> one file.

        tmp -> fsync -> rename -> fsync(dir): a crash must not leave behind a
        truncated artefact that the hash verification later declares valid
        (I-10).
        """
        import os
        h = sha256_bytes(data)
        p = self._blob_path(h)
        if p.exists():
            return h
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        from .osenv import fsync_dir
        fsync_dir(p.parent)
        return h

    def get_blob(self, h: str) -> bytes:
        p = self._blob_path(h)
        data = p.read_bytes()
        if sha256_bytes(data) != h:
            raise IOError(f"artefact {h[:12]} fails hash verification")
        return data

    # ------------------------------------------------------------- models
    def register_model(self, model_hash: str, display_name: str,
                       architecture: str, canonical_map: dict,
                       hash_version: int = MODEL_HASH_VERSION) -> None:
        # hash_version is the last column: SCHEMA creates the five original ones and
        # _upgrade_schema appends it, on a new database and on an old one alike.
        self.db.execute(
            "INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?)",
            (model_hash, display_name, architecture,
             json.dumps(canonical_map, default=str), time.time(), hash_version))
        self.db.commit()

    # ------------------------------------------------- dossier keys across versions
    def _upgrade_schema(self) -> None:
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(models)")}
        if "hash_version" not in cols:
            self.db.execute("ALTER TABLE models ADD COLUMN hash_version INTEGER NOT NULL DEFAULT 1")

    def legacy_models(self) -> list[dict]:
        """Dossiers still keyed by a pre-0.3.1 model hash (header only)."""
        rows = self.db.execute(
            "SELECT model_hash, display_name FROM models WHERE hash_version < ? "
            "ORDER BY display_name", (MODEL_HASH_VERSION,)).fetchall()
        return [{"model_hash": r[0], "display_name": r[1]} for r in rows]

    def _flag_legacy_models(self) -> None:
        """Say so in the inbox, once per dossier (post_inbox de-duplicates), so a
        legacy dossier is never silently different from a new one."""
        for m in self.legacy_models():
            self.post_inbox(
                "migration", m["model_hash"], "dossier",
                f"{m['display_name']}: this dossier is keyed by the pre-0.3.1 model hash "
                f"(header only). It is re-keyed, with its runs, the next time the model "
                f"file is opened.")

    def rekey_model(self, old_hash: str, new_hash: str) -> bool:
        """Carry a dossier keyed by the pre-0.3.1 hash over to the current key.

        Moves the models row (or merges it, if the new key already has one), its
        runs and inbox items, and the harvest folder, which is named by the first
        16 characters of the hash. Sealed artefacts are not touched: they are
        content-addressed and immutable (AZ-4/AZ-5); the mapping is recorded in
        hash_migrations so the old hash inside them can always be resolved.
        Returns True when something was moved."""
        if not old_hash or old_hash == new_hash:
            return False
        db = self.db
        if not db.execute("SELECT 1 FROM models WHERE model_hash=?", (old_hash,)).fetchone():
            return False
        has_new = db.execute("SELECT 1 FROM models WHERE model_hash=?", (new_hash,)).fetchone()
        with db:
            if has_new:
                db.execute("DELETE FROM models WHERE model_hash=?", (old_hash,))
            else:
                db.execute("UPDATE models SET model_hash=?, hash_version=? WHERE model_hash=?",
                           (new_hash, MODEL_HASH_VERSION, old_hash))
            db.execute("UPDATE runs SET model_hash=? WHERE model_hash=?", (new_hash, old_hash))
            db.execute("DELETE FROM inbox WHERE model_hash=? AND kind='migration'", (old_hash,))
            db.execute("UPDATE inbox SET model_hash=? WHERE model_hash=?", (new_hash, old_hash))
            db.execute("INSERT OR REPLACE INTO hash_migrations VALUES (?,?,?)",
                       (old_hash, new_hash, time.time()))
        src = self.root / "harvest" / old_hash[:16]
        dst = self.root / "harvest" / new_hash[:16]
        if src.is_dir() and not dst.exists():
            src.rename(dst)
        return True

    # ------------------------------------------------------------- runs
    def open_run(self, model_hash: str, kind: str, config: dict,
                 seeds: dict | None = None, supersedes: str | None = None) -> RunRecord:
        r = RunRecord(run_id=uuid.uuid4().hex[:16], model_hash=model_hash, kind=kind,
                      config_hash=config_hash(config), seeds=seeds or {},
                      started=time.time(), supersedes=supersedes)
        self.db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (r.run_id, r.model_hash, r.kind, r.config_hash, json.dumps(r.seeds),
             r.started, None, supersedes, 0.0, "working"))
        self.db.commit()
        return r

    def attach(self, r: RunRecord, name: str, data: bytes, media: str = "json") -> str:
        if r.state == "sealed":
            raise SealRefused(f"run {r.run_id} is sealed; a correction is a NEW run")
        h = self.put_blob(data)
        r.artifacts[name] = h
        self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?)",
                        (r.run_id, name, h, len(data), media))
        self.db.commit()
        return h

    # ------------------------------------------------------------- sealing
    def seal(self, r: RunRecord, mission_outcome: dict, mission_spec: dict,
             theta_series: list[float], registry_entries: int,
             registry_baseline: float | None = None,
             pending_priors: list[tuple[str, float]] | None = None) -> GateResult:
        """Three-layer verification that a mission has finished ([PSE-A] 11.4).

          layer 1: the outcome matches the mission specification
          layer 2: the artefacts pass hash verification
          layer 3: the four-condition contribution gate accepts

        Before this moment the run is WORKING STATE and must not appear in the
        dossier.
        """
        missing = [k for k in mission_spec if k not in mission_outcome]
        if missing:
            raise SealRefused(f"layer 1: the outcome does not cover the specification, "
                              f"{missing} missing")
        for k, want in mission_spec.items():
            if want is not None and mission_outcome[k] != want:
                raise SealRefused(
                    f"layer 1: '{k}' = {mission_outcome[k]!r}, the mission requires {want!r}")

        for name, h in r.artifacts.items():
            try:
                self.get_blob(h)
            except Exception as e:
                raise SealRefused(f"layer 2: artefact '{name}' does not verify: {e}")

        gate = contribution_gate(theta_series, registry_entries, registry_baseline)
        if not gate.passed:
            # The em dash and the comma-separated list of condition names are a
            # PARSED FORMAT: the CLI and the web layer split this message on "—"
            # and feed the tail to prognostics.refusal(). Do not reshape it.
            raise SealRefused("layer 3: the contribution gate refuses — "
                              + ", ".join(gate.failed_conditions()))

        r.state = "sealed"
        r.sealed = time.time()
        self.db.execute("UPDATE runs SET state='sealed', sealed=? WHERE run_id=?",
                        (r.sealed, r.run_id))
        # only now does the contribution enter knowledge -- all in one transaction
        if pending_priors:
            self._commit_priors(pending_priors)
        self.db.commit()
        return gate

    # ------------------------------------------------------------- progress
    def progress(self, run_id: str, step: int, total: int, theta: float, label: str) -> None:
        """Working state for the status bar. It is NOT a result and never enters the dossier."""
        self.db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,?,?,?,?)",
                        (run_id, step, total, theta, label, time.time()))
        self.db.commit()

    def active_runs(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT r.run_id, r.kind, m.display_name, r.started FROM runs r "
            "LEFT JOIN models m ON m.model_hash = r.model_hash "
            "WHERE r.state='working' AND r.started > ? ORDER BY r.started DESC",
            (time.time() - 12 * 3600,)).fetchall()
        out = []
        for run_id, kind, name, started in rows:
            pts = self.db.execute("SELECT step, total, theta, label, t FROM progress "
                                  "WHERE run_id=? ORDER BY step", (run_id,)).fetchall()
            if not pts:
                continue
            last = pts[-1]
            out.append({"run_id": run_id, "kind": kind, "model": name, "step": last[0],
                        "total": last[1], "theta": [p[2] for p in pts][-200:],
                        "label": last[3], "age_s": time.time() - last[4]})
        return out

    # ------------------------------------------------------------- death and inbox
    def record_death(self, run_id: str, reason: str, registry: int) -> None:
        """Mission Failure = Death ([PSE §13]): a refused run is RECORDED as dead
        rather than staying "working state" forever. It still does not enter the
        dossier (11.4)."""
        self.db.execute("UPDATE runs SET state='died' WHERE run_id=? AND state='working'",
                        (run_id,))
        self.db.execute("INSERT OR REPLACE INTO deaths VALUES (?,?,?,?)",
                        (run_id, reason, registry, time.time()))
        self.db.commit()

    def post_inbox(self, kind: str, model_hash: str, run_kind: str, text: str) -> str:
        """Decisions the system must not take by itself arrive here as mail, not as
        a popup window ([SPEC] 10.3). The same text is not posted twice while it
        is still unread."""
        dup = self.db.execute("SELECT item_id FROM inbox WHERE model_hash=? AND run_kind=? "
                              "AND text=? AND seen=0", (model_hash, run_kind, text)).fetchone()
        if dup:
            return dup[0]
        iid = uuid.uuid4().hex[:16]
        self.db.execute("INSERT INTO inbox VALUES (?,?,?,?,?,?,0)",
                        (iid, kind, model_hash, run_kind, text, time.time()))
        self.db.commit()
        return iid

    def inbox(self, unseen_only: bool = False) -> list[dict]:
        q = "SELECT item_id, kind, model_hash, run_kind, text, t, seen FROM inbox"
        q += " WHERE seen=0" if unseen_only else ""
        q += " ORDER BY t DESC LIMIT 100"
        return [dict(zip(("item_id", "kind", "model_hash", "run_kind", "text", "t", "seen"), r))
                for r in self.db.execute(q).fetchall()]

    def mark_seen(self, item_id: str) -> None:
        """Read is not the same as deleted: the item stays, it is simply no longer new."""
        self.db.execute("UPDATE inbox SET seen=1 WHERE item_id=?", (item_id,))
        self.db.commit()

    def run_history(self, model_hash: str, kind: str, last: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT run_id, state, started FROM runs WHERE model_hash=? AND kind=? "
            "AND state IN ('sealed','died') ORDER BY started DESC LIMIT ?",
            (model_hash, kind, last)).fetchall()
        return [{"run_id": r[0], "state": r[1], "started": r[2],
                 "registry": self.registry_count(r[0])} for r in rows][::-1]

    # ------------------------------------------------------------- notes
    def add_note(self, run_id: str, author: str, anchor_hash: str, text: str) -> str:
        """AZ-6. There is no update_note and no delete_note -- and there never will be."""
        row = self.db.execute(
            "SELECT 1 FROM artifacts WHERE run_id=? AND content_hash=?",
            (run_id, anchor_hash)).fetchone()
        if not row:
            raise ValueError("a note must be anchored to the hash of an artefact of THIS run")
        if self.db.execute("SELECT state FROM runs WHERE run_id=?",
                           (run_id,)).fetchone()[0] != "sealed":
            raise ValueError("notes can only be attached to a sealed run")
        nid = uuid.uuid4().hex[:16]
        self.db.execute("INSERT INTO notes VALUES (?,?,?,?,?,?)",
                        (nid, run_id, author, anchor_hash, text, time.time()))
        self.db.commit()
        return nid

    # ------------------------------------------------------------- priors
    def prior(self, key: str) -> Welford:
        row = self.db.execute("SELECT n, mean, m2 FROM priors WHERE key=?",
                              (key,)).fetchone()
        return Welford(*row) if row else Welford()

    def test_prior(self, key: str, value: float, z_thresh: float = 2.0,
                   local: "Welford | None" = None) -> tuple[bool, float]:
        """§3.2: would this value MOVE the expectation? READ ONLY.

        The prior is NOT updated here. [PSE §6]: "Before any unit's contribution
        is incorporated into the SHARED_LIB, a four-condition ... check is
        performed". An earlier version updated immediately, so a refused run had
        already written its values in -- repeated refused runs would then
        artificially narrow the variance of the prior. The update goes through
        seal(), after all three layers.

        local: what the unit has learned by itself DURING its own life (not yet
        handed over). Without it, on the first run against a new model every
        measured value is "new", so registry growth is again proportional to
        compute -- exactly the trap from §3.2. The unit learns from itself;
        SHARED_LIB learns from the unit only after the gate.
        """
        w = self.prior(key)
        if local is not None:
            w = Welford.merge(w, local)
        if w.n < 3:
            return True, float("inf")
        sd = w.std
        z = abs(value - w.mean) / sd if sd > 0 else float("inf")
        return z > z_thresh, z

    def _commit_priors(self, pending: list[tuple[str, float]]) -> None:
        for key, value in pending:
            w2 = Welford.merge(self.prior(key), Welford().push(value))
            self.db.execute(
                "INSERT INTO priors VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "n=excluded.n, mean=excluded.mean, m2=excluded.m2, updated=excluded.updated",
                (key, w2.n, w2.mean, w2.m2, time.time()))

    def test_and_update_prior(self, key: str, value: float, z_thresh: float = 2.0
                              ) -> tuple[bool, float]:
        """Test + an immediate write. ONLY for an already accepted contribution (e.g. the tests)."""
        r = self.test_prior(key, value, z_thresh)
        self._commit_priors([(key, value)])
        self.db.commit()
        return r

    def record_registry(self, run_id: str, kind: str, prior_key: str,
                        value: float, z: float) -> None:
        self.db.execute("INSERT INTO registry VALUES (?,?,?,?,?,?,?)",
                        (uuid.uuid4().hex[:16], run_id, kind, prior_key,
                         value, z, time.time()))
        self.db.commit()

    def registry_count(self, run_id: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM registry WHERE run_id=?",
                               (run_id,)).fetchone()[0]

    def registry_baseline(self, model_hash: str, kind: str, last: int = 5) -> float | None:
        rows = self.db.execute(
            "SELECT r.run_id FROM runs r WHERE r.model_hash=? AND r.kind=? "
            "AND r.state='sealed' ORDER BY r.sealed DESC LIMIT ?",
            (model_hash, kind, last)).fetchall()
        if not rows:
            return None
        counts = [self.registry_count(x[0]) for x in rows]
        return sum(counts) / len(counts)

    # ------------------------------------------------------------- dossier
    def dossier(self, model_hash: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT run_id, kind, config_hash, started, sealed, supersedes, state "
            "FROM runs WHERE model_hash=? AND state='sealed' ORDER BY sealed",
            (model_hash,)).fetchall()
        out = []
        for run_id, kind, ch, st, se, sup, state in rows:
            arts = self.db.execute(
                "SELECT name, content_hash, n_bytes FROM artifacts WHERE run_id=?",
                (run_id,)).fetchall()
            notes = self.db.execute(
                "SELECT author, text, created FROM notes WHERE run_id=? ORDER BY created",
                (run_id,)).fetchall()
            out.append({"run_id": run_id, "kind": kind, "config_hash": ch,
                        "started": st, "sealed": se, "supersedes": sup,
                        "artifacts": {a[0]: a[1] for a in arts},
                        "n_registry": self.registry_count(run_id),
                        "notes": [{"author": n[0], "text": n[1]} for n in notes]})
        return out

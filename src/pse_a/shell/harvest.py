"""Harvesting activations -- [SPEC] Sec 5, with the I-9, I-10 and I-11 corrections.

Three faults that the [SPEC] pseudocode commits in a single line (§6 of the
plan, phase 5):
  1. appending to a shared per-layer file -> a crash in the middle of an append
     destroys the earlier good chunks as well (this violates AZ-13: "a crash
     loses at most one chunk")
  2. a buffered write followed by mark_done -> the manifest says "done" while
     the file is truncated
  3. Welford in place -> the baseline and the set of chunks drift apart after a
     crash

Hence:
  - ONE FILE PER (layer, chunk); never an append
  - tmp -> fsync(file) -> rename -> fsync(directory) -> mark_done
  - every chunk carries ITS OWN Welford triple (n, mean, M2); the global
    baseline is DERIVED by a pairwise merge over the done set from the
    manifest (I-9)
  - chunk_id = hash(corpus_hash, offset), not a counter: a change to the corpus
    does not mix old and new activations
  - crc32 in the header; a read refuses a damaged chunk
  - the free space is checked BEFORE every chunk: a full disk in the middle of
    a harvest is a clean death with a log entry, not a truncated file
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"PSEACT1\0"
HEADER = struct.Struct("<8s32sIIIIII")   # magic, model_hash[:32], layer, chunk_no, n, d, dtype, crc
DT_F16, DT_F32 = 1, 2


def chunk_id(corpus_hash: str, offset: int) -> str:
    return hashlib.sha256(f"{corpus_hash}:{offset}".encode()).hexdigest()[:16]


@dataclass
class VWelford:
    """Vector Welford: one (mean, M2) per dimension, merged following Chan."""
    n: int
    mean: np.ndarray
    m2: np.ndarray

    @staticmethod
    def of(x: np.ndarray) -> "VWelford":
        x = x.astype(np.float64)
        mu = x.mean(0)
        return VWelford(x.shape[0], mu, ((x - mu) ** 2).sum(0))

    @staticmethod
    def merge(a: "VWelford", b: "VWelford") -> "VWelford":
        if a.n == 0:
            return b
        if b.n == 0:
            return a
        n = a.n + b.n
        d = b.mean - a.mean
        return VWelford(n, a.mean + d * b.n / n, a.m2 + b.m2 + d * d * a.n * b.n / n)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.m2 / max(self.n - 1, 1))


class HarvestStore:
    def __init__(self, root: str | Path, model_hash: str):
        self.root = Path(root) / "harvest" / model_hash[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_hash = model_hash
        self.db = sqlite3.connect(self.root / "manifest.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks (layer INTEGER, cid TEXT, "
                        "n INTEGER, path TEXT, crc INTEGER, done REAL, "
                        "PRIMARY KEY (layer, cid))")
        self.db.commit()

    # --------------------------------------------------------------- writing
    def _path(self, layer: int, cid: str) -> Path:
        d = self.root / f"L{layer:03d}"
        d.mkdir(exist_ok=True)
        return d / f"{cid}.act"

    def is_done(self, layer: int, cid: str) -> bool:
        return self.db.execute("SELECT 1 FROM chunks WHERE layer=? AND cid=?",
                               (layer, cid)).fetchone() is not None

    def write(self, layer: int, cid: str, chunk_no: int, acts: np.ndarray,
              min_free_bytes: int = 2 << 30) -> None:
        free = shutil.disk_usage(self.root).free
        if free < min_free_bytes + acts.nbytes * 2:
            raise OSError(f"not enough space ({free/1e9:.1f} GB) -- the harvest stops "
                          f"cleanly, with no truncated chunk")
        a16 = np.ascontiguousarray(acts, dtype=np.float16)
        w = VWelford.of(acts)
        body = (a16.tobytes() + np.asarray([w.n], np.int64).tobytes()
                + w.mean.astype(np.float64).tobytes() + w.m2.astype(np.float64).tobytes())
        crc = zlib.crc32(body) & 0xFFFFFFFF
        head = HEADER.pack(MAGIC, self.model_hash[:32].encode(), layer, chunk_no,
                           a16.shape[0], a16.shape[1], DT_F16, crc)
        p = self._path(layer, cid)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(head)
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)                                   # atomic on the same fs
        from .osenv import fsync_dir
        fsync_dir(p.parent)                                  # the step everyone forgets
        self.db.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                        (layer, cid, int(a16.shape[0]), str(p), crc, time.time()))
        self.db.commit()                                     # ONLY NOW is it "done"

    # --------------------------------------------------------------- reading
    def read(self, layer: int, cid: str) -> tuple[np.ndarray, VWelford]:
        p = self._path(layer, cid)
        raw = p.read_bytes()
        magic, mh, L, _, n, d, dt, crc = HEADER.unpack_from(raw, 0)
        if magic != MAGIC:
            raise IOError(f"{p.name}: not a PSE-A chunk")
        body = raw[HEADER.size:]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            raise IOError(f"{p.name}: the crc does not match -- the chunk is damaged and not used")
        na = n * d * 2
        acts = np.frombuffer(body[:na], dtype=np.float16).reshape(n, d)
        off = na
        wn = int(np.frombuffer(body[off:off + 8], np.int64)[0]); off += 8
        mean = np.frombuffer(body[off:off + 8 * d], np.float64); off += 8 * d
        m2 = np.frombuffer(body[off:off + 8 * d], np.float64)
        return acts, VWelford(wn, mean.copy(), m2.copy())

    def write_tokens(self, cid: str, ids: np.ndarray) -> None:
        """The tokens of a chunk (after the stride) -- so that a feature card can
        show WHERE it fires. The same atomic ordering as for the activations."""
        p = self.root / "tokens" / f"{cid}.tok"
        p.parent.mkdir(exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(np.asarray(ids, dtype=np.uint32).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def read_tokens(self, cid: str) -> np.ndarray:
        return np.frombuffer((self.root / "tokens" / f"{cid}.tok").read_bytes(), dtype=np.uint32)

    def done_chunks(self, layer: int) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT cid FROM chunks WHERE layer=? ORDER BY done", (layer,)).fetchall()]

    def baseline(self, layer: int) -> VWelford:
        """I-9: DERIVED from the per-chunk triples over the done set -- always
        exactly consistent with what is on disk."""
        acc = VWelford(0, np.zeros(1), np.zeros(1))
        for cid in self.done_chunks(layer):
            _, w = self.read(layer, cid)
            acc = VWelford.merge(acc, w)
        return acc


def run_harvest(rt, store: HarvestStore, corpus_ids: list[int], corpus_hash: str,
                layers: list[int], chunk_tokens: int, position_stride: int = 1,
                budget_chunks: int | None = None, progress=print) -> dict:
    """[SPEC] HarvestJob. It resumes where it stopped: done chunks are skipped."""
    offsets = list(range(0, len(corpus_ids), chunk_tokens))
    done = skipped = 0
    for no, off in enumerate(offsets):
        if budget_chunks is not None and done >= budget_chunks:
            break
        cid = chunk_id(corpus_hash, off)
        if all(store.is_done(L, cid) for L in layers):
            skipped += 1
            continue
        ids = corpus_ids[off:off + chunk_tokens]
        if len(ids) < 2:
            continue
        acts = rt.forward_with_hooks(ids, {f"resid_pre.{L}" for L in layers})
        store.write_tokens(cid, np.asarray(ids)[::position_stride])
        for L in layers:
            a = acts[f"resid_pre.{L}"].float().numpy()[::position_stride]
            store.write(L, cid, no, a)
        done += 1
        if progress:
            progress(f"  chunk {no+1}/{len(offsets)}  ({len(ids)} tokens, {len(layers)} layers)")
    return {"written": done, "skipped_done": skipped, "total_chunks": len(offsets)}

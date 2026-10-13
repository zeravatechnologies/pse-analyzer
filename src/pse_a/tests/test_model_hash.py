"""model_hash (0.3.1): two models of one geometry must not share a dossier, and a
dossier keyed the old way is carried over once, with everything that hangs off it."""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

from pse_a.core.source.gguf import MODEL_HASH_VERSION, open_gguf
from pse_a.shell.shared_lib import SharedLib

TENSORS = [("blk.0.attn_q.weight", (64, 64)), ("blk.0.ffn_down.weight", (64, 128))]


def write_gguf(path: Path, tensors, seed: int) -> None:
    """A minimal GGUF v3: one metadata key, F32 tensors, 32-byte alignment.
    The same tensors and a different seed give files of identical geometry and
    size whose weights differ: exactly the pair the old key could not tell apart."""
    import numpy as np
    out = bytearray()

    def u32(v): out.extend(struct.pack("<I", v))
    def u64(v): out.extend(struct.pack("<Q", v))
    def s(v):
        b = v.encode(); u64(len(b)); out.extend(b)

    out.extend(b"GGUF"); u32(3); u64(len(tensors)); u64(1)
    s("general.architecture"); u32(8); s("test")           # 8 = STRING
    off, sizes = 0, []
    for name, shape in tensors:
        s(name); u32(len(shape))
        for d in shape:
            u64(d)
        u32(0); u64(off)                                     # 0 = F32
        n = 1
        for d in shape:
            n *= d
        sizes.append(n); off += n * 4
    out.extend(b"\0" * ((-len(out)) % 32))
    rng = np.random.default_rng(seed)
    for n in sizes:
        out.extend(rng.standard_normal(n).astype("<f4").tobytes())
    path.write_bytes(out)


# ---------------------------------------------------------------- the key itself
def test_same_geometry_different_weights_get_different_keys(tmp_path):
    a, b, c = (tmp_path / f"{n}.gguf" for n in "abc")
    write_gguf(a, TENSORS, seed=1)
    write_gguf(b, TENSORS, seed=2)
    write_gguf(c, TENSORS, seed=1)
    assert a.stat().st_size == b.stat().st_size
    ga, gb, gc = open_gguf(a), open_gguf(b), open_gguf(c)
    assert ga.model_hash_legacy() == gb.model_hash_legacy()     # the pre-0.3.1 collision
    assert ga.model_hash() != gb.model_hash()                   # fixed
    assert ga.model_hash() == gc.model_hash()                   # same bytes, same key
    assert ga.model_hash() != ga.model_hash_legacy()
    assert len(ga.model_hash()) == 64


def test_full_hash_is_stable_and_distinct_from_quick(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, TENSORS, seed=3)
    quick, full1, full2 = open_gguf(p).model_hash(), open_gguf(p).model_hash(quick=False), open_gguf(p).model_hash(quick=False)
    assert full1 == full2 and full1 != quick


# ---------------------------------------------------------------- carrying dossiers over
def test_rekey_carries_the_dossier_over(tmp_path):
    root = tmp_path / "data"
    old, new = "a" * 64, "b" * 64
    lib = SharedLib(root)
    try:
        lib.register_model(old, "m.gguf", "qwen2", {}, hash_version=1)
        run = lib.open_run(old, "l1_sweep", {"kind": "l1_sweep"}, {"seed": 0})
        lib.post_inbox("prognoza", old, "l1_sweep", "hello")
        (root / "harvest" / old[:16]).mkdir(parents=True)
    finally:
        lib.close()

    lib = SharedLib(root)                         # a fresh open flags the legacy dossier
    try:
        assert lib.legacy_models() == [{"model_hash": old, "display_name": "m.gguf"}]
        assert any(i["kind"] == "migration" and i["model_hash"] == old for i in lib.inbox())

        assert lib.rekey_model(old, new) is True
        assert lib.legacy_models() == []
        assert lib.db.execute("SELECT hash_version FROM models WHERE model_hash=?", (new,)).fetchone()[0] == MODEL_HASH_VERSION
        assert lib.db.execute("SELECT model_hash FROM runs WHERE run_id=?", (run.run_id,)).fetchone()[0] == new
        assert not any(i["kind"] == "migration" for i in lib.inbox())
        assert any(i["model_hash"] == new and i["text"] == "hello" for i in lib.inbox())
        assert (root / "harvest" / new[:16]).is_dir() and not (root / "harvest" / old[:16]).exists()
        assert lib.db.execute("SELECT new_hash FROM hash_migrations WHERE old_hash=?", (old,)).fetchone()[0] == new
        assert lib.rekey_model(old, new) is False                 # nothing left under the old key
        assert lib.rekey_model(new, new) is False
    finally:
        lib.close()


def test_rekey_merges_when_both_keys_exist(tmp_path):
    root = tmp_path / "data"
    old, new = "c" * 64, "d" * 64
    lib = SharedLib(root)
    try:
        lib.register_model(old, "m.gguf", "qwen2", {}, hash_version=1)
        lib.register_model(new, "m.gguf", "qwen2", {})
        lib.open_run(old, "l1_sweep", {"kind": "l1_sweep"}, {"seed": 0})
        lib.open_run(new, "l1_sweep", {"kind": "l1_sweep"}, {"seed": 1})
        assert lib.rekey_model(old, new) is True
        assert [r[0] for r in lib.db.execute("SELECT model_hash FROM models").fetchall()] == [new]
        assert lib.db.execute("SELECT COUNT(*) FROM runs WHERE model_hash=?", (new,)).fetchone()[0] == 2
    finally:
        lib.close()


def run() -> bool:
    """For `pse-a selftest`: the same checks, pass/fail."""
    good = True
    with tempfile.TemporaryDirectory() as d:
        for fn in (test_same_geometry_different_weights_get_different_keys,
                   test_full_hash_is_stable_and_distinct_from_quick,
                   test_rekey_carries_the_dossier_over,
                   test_rekey_merges_when_both_keys_exist):
            sub = Path(d) / fn.__name__
            sub.mkdir()
            try:
                fn(sub)
                print(f"OK    {fn.__name__}")
            except Exception as e:                            # report, do not stop
                good = False
                print(f"FAIL  {fn.__name__}: {e}")
    return good

"""GGUF reader -- header, metadata, tensor info.

AZ-1: clean core. No state, no theta, no SHARED_LIB.
I-5:  dequantisation strictly on demand -- this module only maps the file and
      reads individual tensors when asked for them. Never the whole model in RAM.
I-8:  the ggml_type is kept PER TENSOR and goes into the report as a covariate.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"

# model_hash: 2 = header + data samples (0.3.1). 1 = header only (before 0.3.1).
MODEL_HASH_VERSION = 2
_SAMPLE_TENSOR = 4096          # bytes read from the start of every tensor
_SAMPLE_WINDOW = 65536         # bytes read at the middle and the end of the data section

# ---------------------------------------------------------------- GGUF KV types
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALAR = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

# ---------------------------------------------------------------- ggml types
# name, block_size (elements per block), type_size (bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0:  ("F32",     1,   4),
    1:  ("F16",     1,   2),
    2:  ("Q4_0",   32,  18),
    3:  ("Q4_1",   32,  20),
    6:  ("Q5_0",   32,  22),
    7:  ("Q5_1",   32,  24),
    8:  ("Q8_0",   32,  34),
    9:  ("Q8_1",   32,  36),
    10: ("Q2_K",  256,  84),
    11: ("Q3_K",  256, 110),
    12: ("Q4_K",  256, 144),
    13: ("Q5_K",  256, 176),
    14: ("Q6_K",  256, 210),
    15: ("Q8_K",  256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS",  256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S",   256, 50),
    20: ("IQ4_NL",   32, 18),
    21: ("IQ3_S",   256, 110),
    22: ("IQ2_S",   256, 82),
    23: ("IQ4_XS",  256, 136),
    24: ("I8",      1,   1),
    25: ("I16",     1,   2),
    26: ("I32",     1,   4),
    27: ("I64",     1,   8),
    28: ("F64",     1,   8),
    29: ("IQ1_M",  256,  56),
    30: ("BF16",    1,   2),
}


def ggml_type_name(t: int) -> str:
    return GGML_TYPES.get(t, (f"UNKNOWN_{t}", 1, 1))[0]


# ---------------------------------------------------------------- structures
@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]      # GGUF order: shape[0] is the fastest dimension
    ggml_type: int
    offset: int                 # relative to the start of the data section

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.ggml_type)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        _, blk, tsz = GGML_TYPES[self.ggml_type]
        return self.n_elements // blk * tsz

    @property
    def is_2d(self) -> bool:
        return len(self.shape) == 2

    def __repr__(self) -> str:
        s = "x".join(str(d) for d in self.shape)
        return f"<{self.name} {s} {self.type_name}>"


@dataclass
class GGUFFile:
    """A mapped GGUF file. Holds metadata only -- no tensor is loaded."""
    path: Path
    version: int
    metadata: dict
    tensors: dict[str, TensorInfo]
    data_offset: int
    alignment: int
    _file_size: int = 0
    _model_hash: str | None = field(default=None, repr=False)

    # ---------------------------------------------------------- identity
    def model_hash(self, quick: bool = True) -> str:
        """Identity of the model: the dossier key (AZ-4).

        quick=True, the default, hashes the header -- GGUF version, file size and
        every tensor's name, shape, type and offset -- PLUS fixed samples of the
        tensor data: the first 4 KiB of every tensor and two 64 KiB windows, at
        the middle and at the end of the data section. A few MB of reads, well
        under a second on an SSD. The samples are what tell apart two models that
        share one geometry and one size: two checkpoints of the same fine-tune,
        two siblings at the same quantisation. Before 0.3.1 the quick hash covered
        the header only, so such pairs collided into one dossier; see
        model_hash_legacy() and SharedLib.rekey_model() for how old dossiers are
        carried over.

        quick=False hashes the whole file after the same header fields. Over 15 GB
        that takes minutes; it is for an explicit audit, not for the key."""
        if self._model_hash:
            return self._model_hash
        h = hashlib.sha256()
        h.update(b"pse-a model_hash v2\n")
        self._hash_header(h)
        with open(self.path, "rb") as f:
            if quick:
                for name in sorted(self.tensors):
                    t = self.tensors[name]
                    f.seek(self.data_offset + t.offset)
                    h.update(f.read(min(t.n_bytes, _SAMPLE_TENSOR)))
                data_len = max(0, self._file_size - self.data_offset)
                for pos in (data_len // 2, max(0, data_len - _SAMPLE_WINDOW)):
                    f.seek(self.data_offset + pos)
                    h.update(f.read(_SAMPLE_WINDOW))
            else:
                while chunk := f.read(1 << 22):
                    h.update(chunk)
        self._model_hash = h.hexdigest()
        return self._model_hash

    def model_hash_legacy(self) -> str:
        """The dossier key used before 0.3.1: header only, no tensor data, so two
        models with the same geometry and size share it. Kept so that an old
        dossier can be found and re-keyed. Never use it for a new dossier."""
        h = hashlib.sha256()
        self._hash_header(h)
        return h.hexdigest()

    def _hash_header(self, h) -> None:
        h.update(str(self.version).encode())
        h.update(str(self._file_size).encode())
        for name in sorted(self.tensors):
            t = self.tensors[name]
            h.update(f"{name}|{t.shape}|{t.ggml_type}|{t.offset}".encode())

    # ---------------------------------------------------------- inventory
    def type_inventory(self) -> dict[str, dict]:
        """I-8: how many tensors and how many parameters per ggml_type.
        This is the proof that Q4_K_M is not one format but a mixture."""
        inv: dict[str, dict] = {}
        for t in self.tensors.values():
            e = inv.setdefault(t.type_name, {"tensors": 0, "elements": 0, "bytes": 0})
            e["tensors"] += 1
            e["elements"] += t.n_elements
            e["bytes"] += t.n_bytes
        return dict(sorted(inv.items(), key=lambda kv: -kv[1]["elements"]))

    def matrices_2d(self) -> list[TensorInfo]:
        """2-D tensors only -- they are the only ones L1 looks at ([SPEC] Sec 4)."""
        return [t for t in self.tensors.values() if t.is_2d]

    def raw_block_data(self, name: str) -> np.ndarray:
        """The raw bytes of one tensor (without dequantisation). I-5."""
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.data_offset + t.offset)
            buf = f.read(t.n_bytes)
        if len(buf) != t.n_bytes:
            raise IOError(
                f"{self.path.name}: tensor {name} needs {t.n_bytes} B, read {len(buf)}. "
                f"The file is truncated or damaged."
            )
        return np.frombuffer(buf, dtype=np.uint8)


# ---------------------------------------------------------------- parser
class _Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise IOError(f"unexpected end of file (wanted {n} B, got {len(b)})")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > (1 << 24):
            raise ValueError(f"unreasonably long string length: {n}")
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            fmt, size = _SCALAR[vtype]
            return struct.unpack(fmt, self.raw(size))[0]
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype = self.u32()
            n = self.u64()
            if etype == STRING:
                return [self.string() for _ in range(n)]
            if etype in _SCALAR:
                fmt, size = _SCALAR[etype]
                buf = self.raw(size * n)
                return list(struct.unpack("<" + fmt[1] * n, buf))
            raise ValueError(f"unsupported array element type: {etype}")
        raise ValueError(f"unknown GGUF value type: {vtype}")


def open_gguf(path: str | Path) -> GGUFFile:
    """Open a GGUF and build the map. Reads no tensor at all (I-5)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.raw(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path.name}: not a GGUF (magic={magic!r})")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"{path.name}: unsupported GGUF version {version}")
        n_tensors = r.u64()
        n_kv = r.u64()

        metadata: dict = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            metadata[key] = r.value(vtype)

        tensors: dict[str, TensorInfo] = {}
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            shape = tuple(r.u64() for _ in range(n_dims))
            ttype = r.u32()
            offset = r.u64()
            if ttype not in GGML_TYPES:
                raise ValueError(f"{path.name}: unknown ggml_type {ttype} for {name}")
            _, blk, _ = GGML_TYPES[ttype]
            n_el = 1
            for d in shape:
                n_el *= d
            if n_el % blk:
                raise ValueError(
                    f"{path.name}: {name} has {n_el} elements, not divisible by block {blk}"
                )
            tensors[name] = TensorInfo(name, shape, ttype, offset)

        alignment = int(metadata.get("general.alignment", 32))
        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment

    # AZ-2 in miniature: a fault that is knowable at open time must not surface later
    for t in tensors.values():
        end = data_offset + t.offset + t.n_bytes
        if end > size:
            raise IOError(
                f"{path.name}: tensor {t.name} ends at {end} B, but the file has {size} B. "
                f"The file is truncated -- check the size of the download."
            )

    return GGUFFile(path, version, metadata, tensors, data_offset, alignment, size)

"""Dequantisation of GGUF blocks -> fp32.

Vectorised through numpy: there is no Python loop over blocks. On an 18944x3584
tensor a loop over blocks would take minutes instead of tenths of a second.

Reference for all formats: ggml-quants.c, the dequantize_row_* functions.
Every format is transcribed exactly, including the order of the nibbles (the
low nibble goes into the first half of the block for Q4_0/Q5_0/Q5_1, but NOT
for the K-quantisations).

The K2 gate: the result must correlate > 0.99 with the same tensor at a higher
precision. A silent error in the bit packing yields a convincing but false
spectrum.
"""
from __future__ import annotations

import numpy as np

from .gguf import GGML_TYPES, GGUFFile, TensorInfo


def _f16(b: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """fp16 scalar per block -> (nb, 1) fp32."""
    return np.ascontiguousarray(b[:, lo:hi]).view(np.float16).astype(np.float32)


# ---------------------------------------------------------------- non-blocked
def _dq_f32(raw, n):
    return raw.view(np.float32)[:n].astype(np.float32, copy=True)


def _dq_f16(raw, n):
    return raw.view(np.float16)[:n].astype(np.float32)


def _dq_bf16(raw, n):
    u = raw.view(np.uint16)[:n].astype(np.uint32) << 16
    return u.view(np.float32)


# ---------------------------------------------------------------- legacy, block 32
def _dq_q8_0(raw, n):
    nb = n // 32
    b = raw.reshape(nb, 34)
    d = _f16(b, 0, 2)
    q = np.ascontiguousarray(b[:, 2:34]).view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)[:n]


def _dq_q4_0(raw, n):
    nb = n // 32
    b = raw.reshape(nb, 18)
    d = _f16(b, 0, 2)
    qs = b[:, 2:18]
    out = np.empty((nb, 32), np.float32)
    out[:, :16] = d * ((qs & 0x0F).astype(np.float32) - 8)
    out[:, 16:] = d * ((qs >> 4).astype(np.float32) - 8)
    return out.reshape(-1)[:n]


def _dq_q5_0(raw, n):
    nb = n // 32
    b = raw.reshape(nb, 22)
    d = _f16(b, 0, 2)
    qh = np.ascontiguousarray(b[:, 2:6]).view(np.uint32).reshape(nb, 1)
    qs = b[:, 6:22].astype(np.uint16)
    j = np.arange(16, dtype=np.uint32)
    xh0 = ((qh >> j) << 4) & 0x10
    xh1 = (qh >> (j + 12)) & 0x10
    out = np.empty((nb, 32), np.float32)
    out[:, :16] = d * (((qs & 0x0F) | xh0).astype(np.float32) - 16)
    out[:, 16:] = d * (((qs >> 4) | xh1).astype(np.float32) - 16)
    return out.reshape(-1)[:n]


def _dq_q5_1(raw, n):
    nb = n // 32
    b = raw.reshape(nb, 24)
    d = _f16(b, 0, 2)
    m = _f16(b, 2, 4)
    qh = np.ascontiguousarray(b[:, 4:8]).view(np.uint32).reshape(nb, 1)
    qs = b[:, 8:24].astype(np.uint16)
    j = np.arange(16, dtype=np.uint32)
    xh0 = ((qh >> j) << 4) & 0x10
    xh1 = (qh >> (j + 12)) & 0x10
    out = np.empty((nb, 32), np.float32)
    out[:, :16] = d * ((qs & 0x0F) | xh0).astype(np.float32) + m
    out[:, 16:] = d * ((qs >> 4) | xh1).astype(np.float32) + m
    return out.reshape(-1)[:n]


def _dq_q4_1(raw, n):
    nb = n // 32
    b = raw.reshape(nb, 20)
    d = _f16(b, 0, 2)
    m = _f16(b, 2, 4)
    qs = b[:, 4:20]
    out = np.empty((nb, 32), np.float32)
    out[:, :16] = d * (qs & 0x0F).astype(np.float32) + m
    out[:, 16:] = d * (qs >> 4).astype(np.float32) + m
    return out.reshape(-1)[:n]


# ---------------------------------------------------------------- K-quantisations, block 256
def _scale_min_k4(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """get_scale_min_k4: 12 bytes -> 8 scales (6 bits) + 8 minima (6 bits).

    This is the easiest place to be off by one shift, and the error is silent.
    """
    nb = sc.shape[0]
    scales = np.empty((nb, 8), np.float32)
    mins = np.empty((nb, 8), np.float32)
    for j in range(8):
        if j < 4:
            s = sc[:, j] & 63
            m = sc[:, j + 4] & 63
        else:
            s = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            m = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        scales[:, j] = s
        mins[:, j] = m
    return scales, mins


def _dq_q4_k(raw, n):
    nb = n // 256
    b = raw.reshape(nb, 144)
    d = _f16(b, 0, 2)
    dmin = _f16(b, 2, 4)
    scales, mins = _scale_min_k4(b[:, 4:16].astype(np.uint16))
    qs = b[:, 16:144]
    out = np.empty((nb, 256), np.float32)
    for k in range(4):                      # j = 0, 64, 128, 192
        q = qs[:, k * 32:(k + 1) * 32]
        d1 = d * scales[:, 2 * k:2 * k + 1]
        m1 = dmin * mins[:, 2 * k:2 * k + 1]
        d2 = d * scales[:, 2 * k + 1:2 * k + 2]
        m2 = dmin * mins[:, 2 * k + 1:2 * k + 2]
        out[:, k * 64:k * 64 + 32] = d1 * (q & 0x0F).astype(np.float32) - m1
        out[:, k * 64 + 32:k * 64 + 64] = d2 * (q >> 4).astype(np.float32) - m2
    return out.reshape(-1)[:n]


def _dq_q5_k(raw, n):
    nb = n // 256
    b = raw.reshape(nb, 176)
    d = _f16(b, 0, 2)
    dmin = _f16(b, 2, 4)
    scales, mins = _scale_min_k4(b[:, 4:16].astype(np.uint16))
    qh = b[:, 16:48].astype(np.uint16)      # 32 bytes, one bit per element
    qs = b[:, 48:176]
    out = np.empty((nb, 256), np.float32)
    for k in range(4):
        q = qs[:, k * 32:(k + 1) * 32].astype(np.uint16)
        u1 = np.uint16(1 << (2 * k))
        u2 = np.uint16(1 << (2 * k + 1))
        d1 = d * scales[:, 2 * k:2 * k + 1]
        m1 = dmin * mins[:, 2 * k:2 * k + 1]
        d2 = d * scales[:, 2 * k + 1:2 * k + 2]
        m2 = dmin * mins[:, 2 * k + 1:2 * k + 2]
        lo = (q & 0x0F) | (((qh & u1) != 0) * 16)
        hi = (q >> 4) | (((qh & u2) != 0) * 16)
        out[:, k * 64:k * 64 + 32] = d1 * lo.astype(np.float32) - m1
        out[:, k * 64 + 32:k * 64 + 64] = d2 * hi.astype(np.float32) - m2
    return out.reshape(-1)[:n]


def _dq_q6_k(raw, n):
    nb = n // 256
    b = raw.reshape(nb, 210)
    ql = b[:, 0:128].astype(np.uint16)
    qh = b[:, 128:192].astype(np.uint16)
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)
    d = _f16(b, 208, 210)
    out = np.empty((nb, 256), np.float32)
    isx = np.arange(32) // 16               # 16x0, 16x1
    for h in range(2):                      # two halves of 128 elements each
        qlh = ql[:, h * 64:(h + 1) * 64]
        qhh = qh[:, h * 32:(h + 1) * 32]
        sch = sc[:, h * 8:(h + 1) * 8]
        lo, hi = qlh[:, :32], qlh[:, 32:64]
        q1 = ((lo & 0x0F) | (((qhh >> 0) & 3) << 4)).astype(np.float32) - 32
        q2 = ((hi & 0x0F) | (((qhh >> 2) & 3) << 4)).astype(np.float32) - 32
        q3 = ((lo >> 4) | (((qhh >> 4) & 3) << 4)).astype(np.float32) - 32
        q4 = ((hi >> 4) | (((qhh >> 6) & 3) << 4)).astype(np.float32) - 32
        o = h * 128
        out[:, o + 0:o + 32] = d * sch[:, isx + 0] * q1
        out[:, o + 32:o + 64] = d * sch[:, isx + 2] * q2
        out[:, o + 64:o + 96] = d * sch[:, isx + 4] * q3
        out[:, o + 96:o + 128] = d * sch[:, isx + 6] * q4
    return out.reshape(-1)[:n]


def _dq_q3_k(raw, n):
    """Q3_K: 16 scales of 6 bits each, EVERY one covering 16 elements (not 32).

    hmask carries the 4th bit: if the bit is set, 0 is subtracted; if it is
    not -- 4. The mask m runs through the whole block (bits 0..7), it is not
    reset between the halves.
    """
    nb = n // 256
    b = raw.reshape(nb, 110)
    hmask = b[:, 0:32].astype(np.uint16)
    qs = b[:, 32:96].astype(np.uint16)
    a = b[:, 96:108].astype(np.uint16)
    d = _f16(b, 108, 110)

    # the aux repacking from ggml-quants.c -- the masks are replicated per byte,
    # so the whole operation is done byte-wise
    s = np.empty((nb, 16), np.float32)
    s[:, 0:4]   = ((a[:, 0:4] & 0x0F) | (((a[:, 8:12] >> 0) & 0x03) << 4)).astype(np.float32)
    s[:, 4:8]   = ((a[:, 4:8] & 0x0F) | (((a[:, 8:12] >> 2) & 0x03) << 4)).astype(np.float32)
    s[:, 8:12]  = ((a[:, 0:4] >> 4)   | (((a[:, 8:12] >> 4) & 0x03) << 4)).astype(np.float32)
    s[:, 12:16] = ((a[:, 4:8] >> 4)   | (((a[:, 8:12] >> 6) & 0x03) << 4)).astype(np.float32)
    s -= 32.0

    out = np.empty((nb, 256), np.float32)
    for half in range(2):                       # n = 0, 128
        qh = qs[:, half * 32:(half + 1) * 32]
        for j in range(4):                      # shift = 0, 2, 4, 6
            shift = 2 * j
            m = np.uint16(1 << (half * 4 + j))
            o = half * 128 + j * 32
            for g in range(2):                  # two groups of 16
                sl = slice(g * 16, g * 16 + 16)
                q = (qh[:, sl] >> shift) & 0x03
                hset = (hmask[:, sl] & m) != 0
                val = q.astype(np.float32) - 4.0 * (~hset)
                dl = d * s[:, half * 8 + j * 2 + g:half * 8 + j * 2 + g + 1]
                out[:, o + g * 16:o + g * 16 + 16] = dl * val
    return out.reshape(-1)[:n]


def _dq_q2_k(raw, n):
    """Q2_K: 16 scale bytes, each = 4 bits of scale + 4 bits of minimum,
    each covering 16 elements."""
    nb = n // 256
    b = raw.reshape(nb, 84)
    sc = b[:, 0:16].astype(np.uint16)
    qs = b[:, 16:80].astype(np.uint16)
    d = _f16(b, 80, 82)
    dmin = _f16(b, 82, 84)

    out = np.empty((nb, 256), np.float32)
    for half in range(2):
        qh = qs[:, half * 32:(half + 1) * 32]
        for j in range(4):
            shift = 2 * j
            o = half * 128 + j * 32
            for g in range(2):
                idx = half * 8 + j * 2 + g
                s1 = sc[:, idx:idx + 1]
                dl = d * (s1 & 0x0F).astype(np.float32)
                ml = dmin * (s1 >> 4).astype(np.float32)
                sl = slice(g * 16, g * 16 + 16)
                q = ((qh[:, sl] >> shift) & 0x03).astype(np.float32)
                out[:, o + g * 16:o + g * 16 + 16] = dl * q - ml
    return out.reshape(-1)[:n]


# the non-linear table for IQ4_NL / IQ4_XS
_KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                      1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def _dq_iq4_nl(raw, n):
    nb = n // 32
    b = raw.reshape(nb, 18)
    d = _f16(b, 0, 2)
    qs = b[:, 2:18]
    out = np.empty((nb, 32), np.float32)
    out[:, :16] = d * _KV_IQ4NL[qs & 0x0F]
    out[:, 16:] = d * _KV_IQ4NL[qs >> 4]
    return out.reshape(-1)[:n]


_DQ = {
    0: _dq_f32, 1: _dq_f16, 30: _dq_bf16,
    2: _dq_q4_0, 3: _dq_q4_1, 6: _dq_q5_0, 7: _dq_q5_1, 8: _dq_q8_0,
    10: _dq_q2_k, 11: _dq_q3_k, 12: _dq_q4_k, 13: _dq_q5_k, 14: _dq_q6_k,
    20: _dq_iq4_nl,
}

SUPPORTED = {GGML_TYPES[t][0] for t in _DQ}


def dequantize(raw: np.ndarray, ggml_type: int, n_elements: int) -> np.ndarray:
    fn = _DQ.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequantisation for {GGML_TYPES.get(ggml_type, ('?',))[0]} is not implemented"
        )
    return fn(raw, n_elements)


def tensor(g: GGUFFile, name: str) -> np.ndarray:
    """One tensor as an fp32 numpy array, shaped (ne1, ne0) = (output, input).

    I-5: only this one tensor is read and dequantised. The caller releases it
    as soon as the ESD has been extracted -- never the whole model in RAM.
    """
    t: TensorInfo = g.tensors[name]
    raw = g.raw_block_data(name)
    flat = dequantize(raw, t.ggml_type, t.n_elements)
    shape = tuple(reversed(t.shape))        # the GGUF ne0 is the fastest dimension
    return flat.reshape(shape)

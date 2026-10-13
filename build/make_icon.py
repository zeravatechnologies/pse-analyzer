"""Generate build/pse-a.ico -- no image library needed, PNG written by hand.

The picture is the thing the tool actually measures: a Marchenko-Pastur bulk
(the eigenvalue density a random matrix would have) with two spike directions
standing outside its right edge. That is the whole L1 story in one glyph.

    .venv\\Scripts\\python.exe build\\make_icon.py
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "pse-a.ico"
SIZES = [16, 24, 32, 48, 64, 128, 256]
SS = 4                                   # supersampling factor
# A thin matrix (lambda small) keeps the MP bulk a rounded hump rather than a
# wall: the 1/x factor piles the density against the left edge as lambda grows.
RATIO = 0.10

BG = np.array([14, 23, 38], float)       # deep slate
BULK = np.array([56, 189, 248], float)   # cyan: the bulk
BULK_HI = np.array([125, 211, 252], float)
SPIKE = np.array([245, 158, 11], float)  # amber: spikes outside the edge
AXIS = np.array([71, 85, 105], float)


def mp_density(x: np.ndarray, ratio: float = RATIO, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density on its support, 0 outside."""
    a = sigma2 * (1 - np.sqrt(ratio)) ** 2
    b = sigma2 * (1 + np.sqrt(ratio)) ** 2
    inside = (x > a) & (x < b)
    out = np.zeros_like(x)
    xs = x[inside]
    out[inside] = np.sqrt((b - xs) * (xs - a)) / (2 * np.pi * sigma2 * ratio * xs)
    return out


def render(n: int) -> np.ndarray:
    """One RGBA frame, n x n, float 0..255."""
    N = n * SS
    img = np.zeros((N, N, 4), float)

    # rounded-square mask
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    r = N * 0.16
    cx = np.clip(xx, r, N - 1 - r)
    cy = np.clip(yy, r, N - 1 - r)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(r + 0.5 - d, 0, 1)

    img[..., :3] = BG
    img[..., 3] = mask * 255

    # plot area
    x0, x1 = N * 0.14, N * 0.86
    ybase, ytop = N * 0.78, N * 0.22
    span = x1 - x0
    bulk_w = span * 0.60               # the bulk gets the left 60%, spikes the rest

    a = (1 - np.sqrt(RATIO)) ** 2
    b = (1 + np.sqrt(RATIO)) ** 2

    def px(v):                          # support value -> pixel column
        return x0 + (v - a) / (b - a) * bulk_w

    eh = ybase - ytop

    # The bulk, binned into bars -- which is how the tool plots an ESD anyway.
    # A filled curve would be truthful too, but MP has a square-root edge, so as
    # a glyph it reads as a wall rather than as a spectrum.
    NB = 7
    edges = np.linspace(a, b, NB + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bars = mp_density(centres)
    bars /= bars.max()

    gap = bulk_w / NB * 0.16
    for i, hfrac in enumerate(bars):
        bx0, bx1 = px(edges[i]) + gap / 2, px(edges[i + 1]) - gap / 2
        htop = ybase - hfrac * eh * 0.86
        cover = (np.clip(xx - bx0 + 1, 0, 1) * np.clip(bx1 - xx + 1, 0, 1)
                 * np.clip(ybase - yy, 0, 1) * np.clip(yy - htop + 1, 0, 1))
        cover = np.where((yy <= ybase) & (yy >= htop - 1), cover, 0) * mask
        # lighter towards the top so the bank of bars has some depth
        t = np.clip((ybase - yy) / eh, 0, 1)[..., None]
        rgb = BULK * (1 - t) + BULK_HI * t
        img[..., :3] = img[..., :3] * (1 - cover[..., None]) + rgb * cover[..., None]

    # the MP edge: a hairline exactly where the bulk must stop
    edge_x = px(b)
    w = max(N * 0.010, 1.0)
    ebar = np.clip(1 - np.abs(xx - edge_x) / w, 0, 1)
    ebar = np.where((yy <= ybase) & (yy >= ybase - eh * 0.30), ebar, 0) * mask
    img[..., :3] = img[..., :3] * (1 - ebar[..., None]) + BULK_HI * ebar[..., None]

    # two spikes standing outside that edge -- the whole point of L1
    for frac, h in ((0.78, 0.74), (0.93, 0.52)):
        sx = x0 + frac * span
        sw = max(N * 0.018, 1.4)
        bar = np.clip(1 - np.abs(xx - sx) / sw, 0, 1)
        bar = np.where((yy <= ybase) & (yy >= ybase - eh * h), bar, 0) * mask
        img[..., :3] = img[..., :3] * (1 - bar[..., None]) + SPIKE * bar[..., None]
        # cap dot
        cyy, cxx = ybase - eh * h, sx
        dd = np.sqrt((xx - cxx) ** 2 + (yy - cyy) ** 2)
        dot = np.clip(max(N * 0.028, 1.6) - dd, 0, 1) * mask
        img[..., :3] = img[..., :3] * (1 - dot[..., None]) + SPIKE * dot[..., None]

    # baseline
    ax = np.clip(1 - np.abs(yy - ybase) / max(N * 0.008, 1.0), 0, 1)
    ax = np.where((xx >= x0 * 0.85) & (xx <= x1 * 1.02), ax, 0) * mask
    img[..., :3] = img[..., :3] * (1 - ax[..., None]) + AXIS * ax[..., None]

    # downsample
    small = img.reshape(n, SS, n, SS, 4).mean(axis=(1, 3))
    return np.clip(small, 0, 255)


def png_bytes(frame: np.ndarray) -> bytes:
    h, w, _ = frame.shape
    data = frame.astype(np.uint8).tobytes()
    rows = b"".join(b"\x00" + data[y * w * 4:(y + 1) * w * 4] for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def main() -> None:
    frames = [(n, png_bytes(render(n))) for n in SIZES]
    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    entries, blobs = b"", b""
    for n, blob in frames:
        entries += struct.pack("<BBBBHHII", n if n < 256 else 0, n if n < 256 else 0,
                               0, 0, 1, 32, len(blob), offset)
        blobs += blob
        offset += len(blob)
    OUT.write_bytes(header + entries + blobs)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, sizes {SIZES})")


if __name__ == "__main__":
    main()

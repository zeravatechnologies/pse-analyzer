"""TorchRuntime for Qwen2 from a DEQUANTISED GGUF -- [SPEC] Sec 2.

"Runtime -- a forward-capable execution engine with hook points and optionally
vector-Jacobian products. Implementation: TorchRuntime (HF checkpoint or
dequantized GGUF)." This is the second variant: the very file L1 reads also
becomes the model that runs, so L1 and L2-L4 speak about the SAME weights.

K1 measured on the Acer (§13.10): torch 2.14 works without AVX2 (the DEFAULT
kernels, ~2.5x slower than numpy's OpenBLAS), autograd works. That is why this
runtime is developed and checked here, while the scale work goes to the Dell.

Details that break silently if you get them wrong:
  - llama.cpp does NOT permute q/k for Qwen2 (it only does that for LLaMA) ->
    RoPE is the HF "rotate_half" (NEOX) form, not interleaved
  - the embeddings are tied: W_U = token_embd (there is no output.weight)
  - Qwen2 has a bias on q, k, v (but not on o)
  - the logit lens: ALWAYS the final norm before W_U ([SPEC] Sec 5)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..source.gguf import open_gguf
from ..source import dequant as dq

QWEN2_PRETOK = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
                r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


@dataclass
class Capabilities:
    has_hooks: bool = True
    has_autograd: bool = True
    device: str = "cpu"
    max_memory: int = 0


def build_tokenizer(meta: dict):
    """Byte-level BPE from the GGUF metadata (vocab + merges + the Qwen2 pre-tokenizer)."""
    from tokenizers import Tokenizer, Regex, AddedToken, models, pre_tokenizers, decoders
    toks = meta["tokenizer.ggml.tokens"]
    types = meta.get("tokenizer.ggml.token_type", [1] * len(toks))
    vocab = {t: i for i, t in enumerate(toks)}
    merges = [tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False,
                              byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_PRETOK), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    specials = [AddedToken(t, special=True, normalized=False)
                for t, ty in zip(toks, types) if ty == 3]      # 3 = control
    tk.add_special_tokens(specials)
    return tk


class Qwen2Runtime:
    def __init__(self, gguf_path, dtype: str = "float32", n_layers: int | None = None):
        import torch
        self.torch = torch
        self.dtype = getattr(torch, dtype)
        g = open_gguf(gguf_path)
        m = g.metadata
        if m.get("general.architecture") != "qwen2":
            raise ValueError("this runtime is for qwen2")
        self.n_layers = int(m["qwen2.block_count"]) if n_layers is None else int(n_layers)
        self.d = int(m["qwen2.embedding_length"])
        self.n_head = int(m["qwen2.attention.head_count"])
        self.n_kv = int(m["qwen2.attention.head_count_kv"])
        self.hd = self.d // self.n_head
        self.eps = float(m["qwen2.attention.layer_norm_rms_epsilon"])
        self.rope_base = float(m["qwen2.rope.freq_base"])
        self.tok = build_tokenizer(m)
        self.caps = Capabilities()

        T = lambda name: torch.from_numpy(np.ascontiguousarray(dq.tensor(g, name))).to(self.dtype)
        self.embed = T("token_embd.weight")                     # (vocab, d); tied = W_U
        self.final_norm_w = T("output_norm.weight")
        self.W_U = T("output.weight") if "output.weight" in g.tensors else self.embed
        self.blocks = []
        for L in range(self.n_layers):
            p = f"blk.{L}."
            b = {}
            for k in ("attn_norm", "ffn_norm"):
                b[k] = T(p + k + ".weight")
            for k in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
                b[k] = T(p + k + ".weight")
            for k in ("attn_q", "attn_k", "attn_v"):
                b[k + "_b"] = T(p + k + ".bias")
            self.blocks.append(b)

    # ------------------------------------------------------------------ helpers
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def _rms(self, x, w):
        t = self.torch
        return x * t.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def final_norm(self, h):
        return self._rms(h, self.final_norm_w)

    def _rope(self, T: int):
        t = self.torch
        inv = 1.0 / (self.rope_base ** (t.arange(0, self.hd, 2, dtype=t.float64) / self.hd))
        f = t.arange(T, dtype=t.float64)[:, None] * inv[None, :]
        emb = t.cat([f, f], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rot_half(x):
        import torch
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _block(self, x, b, cos, sin):
        t = self.torch
        T = x.shape[0]
        h = self._rms(x, b["attn_norm"])
        q = (h @ b["attn_q"].T + b["attn_q_b"]).view(T, self.n_head, self.hd).transpose(0, 1)
        k = (h @ b["attn_k"].T + b["attn_k_b"]).view(T, self.n_kv, self.hd).transpose(0, 1)
        v = (h @ b["attn_v"].T + b["attn_v_b"]).view(T, self.n_kv, self.hd).transpose(0, 1)
        q = q * cos + self._rot_half(q) * sin
        k = k * cos + self._rot_half(k) * sin
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        a = t.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None],
                                                         is_causal=True)[0]
        a = a.transpose(0, 1).reshape(T, self.d)
        x = x + a @ b["attn_output"].T
        h = self._rms(x, b["ffn_norm"])
        mlp = (t.nn.functional.silu(h @ b["ffn_gate"].T) * (h @ b["ffn_up"].T)) @ b["ffn_down"].T
        return x + mlp

    # ------------------------------------------------------------------ [SPEC] Sec 2
    def run_resid(self, ids, start_layer: int = 0, x=None, stop_layer: int | None = None,
                  hook_points=None):
        """The residual stream: resid_pre.L is the INPUT to block L; resid_post.L
        is its output. It can start from an arbitrary layer with a given x
        (used by vjp)."""
        t = self.torch
        ids = t.as_tensor(ids, dtype=t.long)
        if x is None:
            x = self.embed[ids]
        cos, sin = self._rope(x.shape[0])
        stop = self.n_layers if stop_layer is None else stop_layer
        acts = {}
        for L in range(start_layer, stop):
            if hook_points is None or f"resid_pre.{L}" in hook_points:
                acts[f"resid_pre.{L}"] = x
            x = self._block(x, self.blocks[L], cos, sin)
        acts["resid_final"] = x
        return x, acts

    def forward_with_hooks(self, tokens, hook_points=None) -> dict:
        with self.torch.no_grad():
            _, acts = self.run_resid(tokens, hook_points=hook_points)
        return acts

    def logits(self, tokens):
        with self.torch.no_grad():
            x, _ = self.run_resid(tokens)
            return self.final_norm(x) @ self.W_U.T

    def vjp(self, tokens, to_layer: int, cotangents, position: int = -1,
            normed: bool = False, batched: bool = False):
        """The rows of W_U·J_l via a VJP: d(u·h_final[pos]) / d h_l, for every u.

        [SPEC] Sec 7: the full d_model × d_model Jacobian is never materialised.
        Returns (n_cot, T, d): the gradient with respect to the WHOLE residual
        stream at layer to_layer (the dependence on earlier positions is part of
        the answer -- that is causal attention).
        """
        t = self.torch
        with t.no_grad():
            _, acts = self.run_resid(tokens, stop_layer=to_layer + 1,
                                     hook_points={f"resid_pre.{to_layer}"})
            h = acts[f"resid_pre.{to_layer}"].detach()
        h = h.clone().requires_grad_(True)
        xf, _ = self.run_resid(tokens, start_layer=to_layer, x=h)
        out = xf[position]                                   # (d,)
        if normed:                                           # the lens reads THROUGH the final norm
            out = self.final_norm(out)
        U = t.as_tensor(cotangents, dtype=self.dtype)        # (n_cot, d)
        if batched:
            # one batched backward instead of n_cot separate ones (vmap over the cotangents)
            g, = t.autograd.grad(out, h, grad_outputs=U, is_grads_batched=True)
            return g
        grads = []
        for i in range(U.shape[0]):
            g, = t.autograd.grad(out @ U[i], h, retain_graph=i < U.shape[0] - 1)
            grads.append(g)
        return t.stack(grads)


# ---------------------------------------------------------------------- generation
def _block_cached(rt, x, b, cos, sin, cache, L):
    """The same block, but it caches K and V (after RoPE) so the next token does
    not recompute the whole prompt."""
    t = rt.torch
    T = x.shape[0]
    h = rt._rms(x, b["attn_norm"])
    q = (h @ b["attn_q"].T + b["attn_q_b"]).view(T, rt.n_head, rt.hd).transpose(0, 1)
    k = (h @ b["attn_k"].T + b["attn_k_b"]).view(T, rt.n_kv, rt.hd).transpose(0, 1)
    v = (h @ b["attn_v"].T + b["attn_v_b"]).view(T, rt.n_kv, rt.hd).transpose(0, 1)
    q = q * cos + rt._rot_half(q) * sin
    k = k * cos + rt._rot_half(k) * sin
    if L in cache:
        k = t.cat([cache[L][0], k], dim=1)
        v = t.cat([cache[L][1], v], dim=1)
    cache[L] = (k, v)
    rep = rt.n_head // rt.n_kv
    kk, vv = k.repeat_interleave(rep, dim=0), v.repeat_interleave(rep, dim=0)
    S = kk.shape[1]
    # the causal mask: new queries see everything cached + themselves backwards
    mask = t.ones(T, S, dtype=t.bool).tril(diagonal=S - T)
    a = t.nn.functional.scaled_dot_product_attention(q[None], kk[None], vv[None],
                                                     attn_mask=mask[None, None])[0]
    x = x + a.transpose(0, 1).reshape(T, rt.d) @ b["attn_output"].T
    h = rt._rms(x, b["ffn_norm"])
    return x + (t.nn.functional.silu(h @ b["ffn_gate"].T) * (h @ b["ffn_up"].T)) @ b["ffn_down"].T


def generate(rt, ids: list[int], max_new: int = 256, temperature: float = 0.7,
             top_p: float = 0.9, stop_ids: tuple[int, ...] = (), seed: int = 0):
    """A token generator with a KV cache. Yields one id at a time."""
    t = rt.torch
    g = t.Generator().manual_seed(seed)
    cache: dict = {}
    pos = 0
    cur = list(ids)
    cos_all, sin_all = rt._rope(len(ids) + max_new + 1)
    with t.no_grad():
        for _ in range(max_new):
            x = rt.embed[t.as_tensor(cur, dtype=t.long)]
            T = x.shape[0]
            cos, sin = cos_all[pos:pos + T], sin_all[pos:pos + T]
            for L in range(rt.n_layers):
                x = _block_cached(rt, x, rt.blocks[L], cos, sin, cache, L)
            pos += T
            logits = (rt.final_norm(x[-1]) @ rt.W_U.T).float()
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                p = (logits / temperature).softmax(-1)
                sp, si = p.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p
                sp = sp * keep
                nxt = int(si[t.multinomial(sp / sp.sum(), 1, generator=g)])
            if nxt in stop_ids:
                return
            yield nxt
            cur = [nxt]


def chat_prompt(rt, system: str, turns: list[tuple[str, str]]) -> list[int]:
    """The Qwen2 chat template (ChatML) from the special tokens the GGUF already carries."""
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, text in turns:
        s += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    s += "<|im_start|>assistant\n"
    return rt.tok.encode(s, add_special_tokens=False).ids

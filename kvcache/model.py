"""A hand-written forward pass for Qwen2-family models (Qwen2.5-0.5B etc.).

We reuse the HuggingFace weights but re-implement the decoder so that we own the
KV cache: we can store pre-RoPE keys, evict arbitrary entries per head,
and receive the attention probabilities of every layer through a callback.
Correctness is checked against the HF eager implementation in tests/.
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .cache import KVCache, LayerKV
from .rope import apply_rope


@dataclass
class ModelSpec:
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_eps: float
    vocab_size: int


def spec_from_config(cfg) -> ModelSpec:
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:  # newer transformers moves it into rope_parameters
        theta = cfg.rope_parameters["rope_theta"]
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return ModelSpec(
        n_layers=cfg.num_hidden_layers,
        hidden=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        rope_theta=float(theta),
        rms_eps=cfg.rms_norm_eps,
        vocab_size=cfg.vocab_size,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def load_hf(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
    try:
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="eager", local_files_only=offline)
    except TypeError:  # older transformers
        hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager", local_files_only=offline)
    hf = hf.to(device).eval()
    return hf, tok


AttnCallback = Callable[[int, torch.Tensor], None]  # (layer_idx, probs[1,H,L,T])


class ManualQwen2:
    """Wraps an HF Qwen2ForCausalLM and runs its weights with our own forward."""

    def __init__(self, hf_model):
        cfg = hf_model.config
        if cfg.model_type != "qwen2" or getattr(cfg, "rope_scaling", None) or getattr(cfg, "use_sliding_window", False):
            raise ValueError("ManualQwen2 supports Qwen2 with unscaled RoPE and full attention only")
        self.hf = hf_model
        self.spec = spec_from_config(hf_model.config)
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype

    def new_cache(self, rope_mode: str = "recompute") -> KVCache:
        return KVCache(self.spec.n_layers, rope_mode)

    # ------------------------------------------------------------------ attention
    def _attention(self, li: int, layer, x: torch.Tensor, cache: KVCache,
                   attn_callback: Optional[AttnCallback]):
        s = self.spec
        B, L, _ = x.shape
        assert B == 1, "batch size 1 only"
        at = layer.self_attn
        q = at.q_proj(x).view(B, L, s.n_heads, s.head_dim).transpose(1, 2)
        k = at.k_proj(x).view(B, L, s.n_kv_heads, s.head_dim).transpose(1, 2)
        v = at.v_proj(x).view(B, L, s.n_kv_heads, s.head_dim).transpose(1, 2)

        lc: LayerKV = cache.layers[li]
        T_old = lc.T
        orig_pos = torch.arange(cache.n_seen, cache.n_seen + L, device=x.device)

        if cache.rope_mode == "stale":
            # HF-style: rotate with position = cache index at insertion time, cache post-RoPE keys.
            ins_pos = torch.arange(T_old, T_old + L, device=x.device)
            q = apply_rope(q, ins_pos, s.rope_theta)
            k = apply_rope(k, ins_pos, s.rope_theta)
            lc.append(k, v, orig_pos)
            k_all = lc.k
        elif cache.rope_mode == "absolute":
            lc.append(k, v, orig_pos)
            q = apply_rope(q, orig_pos, s.rope_theta)
            k_all = apply_rope(lc.k, lc.pos, s.rope_theta)  # per-head original positions
        else:  # recompute (correct)
            lc.append(k, v, orig_pos)
            T = lc.T
            q = apply_rope(q, torch.arange(T - L, T, device=x.device), s.rope_theta)
            k_all = apply_rope(lc.k, torch.arange(T, device=x.device), s.rope_theta)

        T = lc.T
        G = s.n_heads // s.n_kv_heads
        k_rep = k_all.repeat_interleave(G, dim=1)  # [1,H,T,D]
        v_rep = lc.v.repeat_interleave(G, dim=1)

        # Match HF's scaling in the model dtype, followed by fp32 softmax.
        scores = (torch.matmul(q, k_rep.transpose(-1, -2)) * (s.head_dim ** -0.5)).float()
        # causal mask inside the new chunk: query i (0..L-1) sees cache index j <= T-L+i
        qi = torch.arange(L, device=x.device)[:, None]
        kj = torch.arange(T, device=x.device)[None, :]
        allowed = kj <= (T - L) + qi
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)  # fp32
        lc.add_scores(probs, s.n_kv_heads)
        if attn_callback is not None:
            attn_callback(li, probs)
        out = torch.matmul(probs.to(v_rep.dtype), v_rep)  # [1,H,L,D]
        out = out.transpose(1, 2).reshape(B, L, s.n_heads * s.head_dim)
        return at.o_proj(out)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: KVCache,
                attn_callback: Optional[AttnCallback] = None,
                return_last_only: bool = False) -> torch.Tensor:
        """input_ids: [1, L]. Appends to `cache` and returns logits [1, L, V]
        (or [1, 1, V] if return_last_only)."""
        s = self.spec
        m = self.hf.model
        x = m.embed_tokens(input_ids)
        for li, layer in enumerate(m.layers):
            h = rms_norm(x, layer.input_layernorm.weight, s.rms_eps)
            x = x + self._attention(li, layer, h, cache, attn_callback)
            h = rms_norm(x, layer.post_attention_layernorm.weight, s.rms_eps)
            mlp = layer.mlp
            x = x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))
        cache.n_seen += input_ids.shape[1]
        x = rms_norm(x, m.norm.weight, s.rms_eps)
        if return_last_only:
            x = x[:, -1:]
        return self.hf.lm_head(x)

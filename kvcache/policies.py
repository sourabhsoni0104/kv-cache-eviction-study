"""Eviction policies. Each returns, per KV head, the sorted cache
indices to KEEP so the cache shrinks to `budget` entries.

  sliding   : keep the most recent `budget` tokens. Naive baseline; drops the
              attention sinks -> perplexity explodes (StreamingLLM, Fig. 1).
  streaming : StreamingLLM (Xiao et al. 2023) - keep the first `n_sink` tokens
              plus the most recent `budget - n_sink`.
  h2o       : Heavy-Hitter Oracle (Zhang et al. 2023) - keep the most recent
              `recent` tokens plus, per head, the tokens with the largest
              accumulated attention score among the rest.
"""
from __future__ import annotations
import torch

from .cache import LayerKV


class EvictionPolicy:
    name = "base"

    def keep_indices(self, lc: LayerKV, budget: int) -> torch.Tensor:
        raise NotImplementedError

    def min_retained(self, budget: int) -> int:
        """Old entries that must survive a chunk reservation; caps the chunk size."""
        return 0

    def __repr__(self):
        return self.name


class NoEviction(EvictionPolicy):
    name = "full"

    def keep_indices(self, lc, budget):
        Hkv, T = lc.pos.shape
        return torch.arange(T, device=lc.pos.device).expand(Hkv, T)


class SlidingWindow(EvictionPolicy):
    name = "sliding"

    def keep_indices(self, lc, budget):
        Hkv, T = lc.pos.shape
        budget = min(T, max(0, budget))
        return torch.arange(T - budget, T, device=lc.pos.device).expand(Hkv, budget)


class StreamingLLM(EvictionPolicy):
    def __init__(self, n_sink: int = 4):
        if n_sink < 0:
            raise ValueError("n_sink must be nonnegative")
        self.n_sink = n_sink
        self.name = f"streaming(sink={n_sink})"

    def min_retained(self, budget):
        return self.n_sink

    def keep_indices(self, lc, budget):
        Hkv, T = lc.pos.shape
        dev = lc.pos.device
        budget = min(T, max(0, budget))
        n_sink = min(self.n_sink, budget)
        n_recent = budget - n_sink
        # cache indices 0..n_sink-1 are always the original first tokens because
        # we never evict them.
        idx = torch.cat([torch.arange(n_sink, device=dev),
                         torch.arange(T - n_recent, T, device=dev)])
        return idx.expand(Hkv, budget)


class H2O(EvictionPolicy):
    def __init__(self, recent_ratio: float = 0.5):
        if not 0 <= recent_ratio <= 1:
            raise ValueError("recent_ratio must be between 0 and 1")
        self.recent_ratio = recent_ratio
        self.name = f"h2o(recent={recent_ratio})"

    def min_retained(self, budget):
        # A chunk as large as the budget would evict to zero entries and erase
        # every heavy hitter; keep at least half the budget across reservations.
        return budget // 2

    def keep_indices(self, lc, budget):
        Hkv, T = lc.pos.shape
        dev = lc.pos.device
        budget = min(T, max(0, budget))
        n_recent = min(budget, max(1, int(budget * self.recent_ratio)))
        n_heavy = budget - n_recent
        recent = torch.arange(T - n_recent, T, device=dev).expand(Hkv, n_recent)
        cand_scores = lc.score[:, : T - n_recent]  # [Hkv, T-n_recent]
        heavy = cand_scores.argsort(dim=1, descending=True, stable=True)[:, :n_heavy]
        keep = torch.cat([heavy, recent], dim=1)
        return keep.sort(dim=1).values


def make_policy(name: str, **kw) -> EvictionPolicy:
    name = name.lower()
    if name == "full":
        return NoEviction()
    if name == "sliding":
        return SlidingWindow()
    if name == "streaming":
        return StreamingLLM(n_sink=kw.get("n_sink", 4))
    if name == "h2o":
        return H2O(recent_ratio=kw.get("recent_ratio", 0.5))
    raise ValueError(f"unknown policy {name}")


POLICY_NAMES = ["sliding", "streaming", "h2o"]

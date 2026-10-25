"""Perplexity on a long document under a fixed KV budget.

Reports overall PPL and PPL restricted to tokens *after* the first eviction
(the only region where policies differ from the full cache).

    python -m scripts.eval_ppl --policy sliding --budget 512
    python -m scripts.eval_ppl --policy streaming --budget 512 --rope_mode stale
"""
from __future__ import annotations
import argparse
import math

import torch

from kvcache.policies import make_policy, POLICY_NAMES
from kvcache.streaming import StreamingRunner
from kvcache.utils import build_model, load_text, save_json, DEFAULT_MODEL


def run_ppl(model, ids: torch.Tensor, policy_name: str, budget, rope_mode="recompute",
            chunk_size=32, eval_start=None, layer_budgets=None) -> dict:
    policy = make_policy(policy_name)
    runner = StreamingRunner(model, policy, budget=budget, rope_mode=rope_mode, chunk_size=chunk_size,
                             layer_budgets=layer_budgets)
    nll = runner.feed_nll(ids)  # [N-1]
    # nll[i] predicts token i+1 from query i. The first query computed after
    # eviction has original position first_eviction_seen.
    start = runner.first_eviction_seen if runner.first_eviction_seen is not None else len(nll)
    post = nll[start:] if start < len(nll) else nll[:0]
    if eval_start is None:
        eval_start = 0
    if not 0 <= eval_start < len(nll):
        raise ValueError("eval_start must leave at least one scored prediction")
    return {
        "policy": policy_name, "budget": budget, "rope_mode": rope_mode,
        "n_tokens": int(ids.shape[1]), "chunk_size": chunk_size,
        "ppl_all": float(math.exp(nll.mean())),
        "ppl_post_budget": float(math.exp(post.mean())) if len(post) else None,
        "eval_start": eval_start, "ppl_eval": float(math.exp(nll[eval_start:].mean())),
        "nll_sum_eval": float(nll[eval_start:].double().sum()), "n_eval": len(nll) - eval_start,
        "first_eviction_seen": runner.first_eviction_seen, "n_evictions": runner.n_evictions,
        "peak_kv_bytes": runner.peak_kv_bytes, "peak_cache_tokens": runner.peak_cache_tokens,
        "metadata_bytes": runner.cache.metadata_bytes(),
        "nll_curve": nll.view(-1).cpu().numpy().tolist(),
        "final_cache_T": runner.cache.T,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default="data/pride_and_prejudice.txt")
    ap.add_argument("--n_tokens", type=int, default=6144)
    ap.add_argument("--policy", default="streaming", choices=POLICY_NAMES + ["full"])
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--rope_mode", default="recompute")
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok, _ = build_model(args.model, args.device, args.dtype)
    ids = tok(load_text(args.text), return_tensors="pt").input_ids[:, : args.n_tokens].to(model.device)
    budget = None if args.policy == "full" else args.budget
    res = run_ppl(model, ids, args.policy, budget, args.rope_mode, args.chunk_size)
    print(f"{args.policy:10s} budget={budget} rope={args.rope_mode}: "
          f"PPL all={res['ppl_all']:.3f}  post-budget={res['ppl_post_budget']}")
    out = args.out or f"results/ppl_{args.policy}_{budget}_{args.rope_mode}.json"
    res.pop("nll_curve")
    save_json(res, out)


if __name__ == "__main__":
    main()

"""Does position handling after eviction matter?

Same policy and budget, three ways of handling RoPE:
  recompute (ours, correct) / absolute (original positions) / stale (HF-slicing bug).
Reports perplexity and needle retrieval for a needle that SURVIVES eviction
(depth close to the end, i.e. inside the recent window) so that any failure is
due to positions, not to the fact having been dropped.

    python -m scripts.rope_ablation --policy streaming --budget 512 --ctx 2048
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kvcache.cache import ROPE_MODES
from kvcache.utils import build_model, load_text, save_json, DEFAULT_MODEL
from scripts.eval_ppl import run_ppl
from scripts.eval_niah import run_niah


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default="data/pride_and_prejudice.txt")
    ap.add_argument("--n_tokens", type=int, default=4096)
    ap.add_argument("--policy", default="streaming")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--depths", default="0.9,0.95,0.98")
    ap.add_argument("--n_trials", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--chunk_size", type=int, default=32)
    args = ap.parse_args()

    model, tok, _ = build_model(args.model, args.device, args.dtype)
    ids = tok(load_text(args.text), return_tensors="pt").input_ids[:, : args.n_tokens].to(model.device)
    depths = [float(d) for d in args.depths.split(",")]

    out = {}
    for mode in ROPE_MODES:
        rp = run_ppl(model, ids, args.policy, args.budget, rope_mode=mode,
                     chunk_size=args.chunk_size, eval_start=args.budget)
        rn = run_niah(model, tok, args.policy, args.budget, args.ctx, depths, args.n_trials,
                      rope_mode=mode, chunk_size=args.chunk_size, verbose=False)
        retained = all(t["retention"]["complete_in_all_heads"] for t in rn["trials"])
        if not retained:
            raise ValueError("RoPE ablation is confounded by eviction: move needles closer to the end or raise budget")
        out[mode] = {"ppl_post_budget": rp["ppl_post_budget"], "ppl_all": rp["ppl_all"],
                     "niah_acc": rn["accuracy"], "niah_per_depth": rn["per_depth"],
                     "ppl": rp, "niah": rn, "all_needles_retained": retained}
        print(f"{mode:10s} ppl_post={rp['ppl_post_budget']:.3f}  niah={rn['accuracy']:.2f} {rn['per_depth']}")

    reference = run_niah(model, tok, "full", None, args.ctx, depths, args.n_trials,
                          chunk_size=args.chunk_size, verbose=False)
    save_json({"model": args.model, "dtype": str(model.dtype), "device": str(model.device),
               "policy": args.policy, "budget": args.budget, "ctx": args.ctx,
               "chunk_size": args.chunk_size, "full_niah": reference, "results": out},
              "results/rope_ablation.json")
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(ROPE_MODES, [out[m]["ppl_post_budget"] for m in ROPE_MODES]); axes[0].set_ylabel("perplexity (post-budget)")
    axes[1].bar(ROPE_MODES, [out[m]["niah_acc"] for m in ROPE_MODES]); axes[1].set_ylabel("needle accuracy (needle kept in cache)")
    axes[1].set_ylim(0, 1.05)
    fig.suptitle(f"RoPE handling after eviction - {args.policy}, budget={args.budget}")
    fig.savefig("plots/rope_ablation.png", dpi=130, bbox_inches="tight")
    print("saved plots/rope_ablation.png")


if __name__ == "__main__":
    main()

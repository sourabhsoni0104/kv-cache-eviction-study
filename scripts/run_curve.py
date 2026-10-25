"""Quality vs. memory curve across cache budgets for all three policies.

Quality = perplexity on a long document (post-budget region) and needle-in-a-
haystack accuracy. Memory = KV-cache bytes at the budget (theoretical, exact for
our cache layout). Also runs the full cache as the reference.

    python -m scripts.run_curve --budgets 128,256,512,1024,2048 --n_tokens 6144 --ctx 2048
"""
from __future__ import annotations
import argparse
import os
import math
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from kvcache.policies import POLICY_NAMES
from kvcache.utils import build_model, load_text, save_json, hardware_report, DEFAULT_MODEL
from scripts.eval_ppl import run_ppl
from scripts.eval_niah import run_niah


def kv_bytes(model, T):
    s = model.spec
    return 2 * s.n_layers * s.n_kv_heads * T * s.head_dim * torch.tensor([], dtype=model.dtype).element_size()


def plot(res, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    full = res["full"]
    for p in POLICY_NAMES:
        rows = res["rows"][p]
        mem = [r["kv_mb"] for r in rows]
        axes[0].plot(mem, [r["ppl_eval"] for r in rows], "o-", label=p)
        axes[1].plot(mem, [r["niah_acc"] for r in rows], "o-", label=p)
    axes[0].axhline(full["ppl_eval"], ls="--", c="gray", label=f"full cache ({full['kv_mb']:.0f} MiB)")
    axes[1].axhline(full["niah_acc"], ls="--", c="gray", label="full cache")
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("KV cache memory (MiB)"); axes[0].set_ylabel("perplexity (identical evaluation region)")
    axes[1].set_xlabel("KV cache memory (MiB)"); axes[1].set_ylabel("needle retrieval accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    # Log-scale default ticks overlap at these values; label the measured budgets instead.
    budget_mem = [r["kv_mb"] for r in res["rows"][POLICY_NAMES[0]]]
    for ax in axes:
        ax.set_xticks(budget_mem, labels=[f"{m:g}" for m in budget_mem])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Quality vs. KV-cache memory")
    fig.savefig(os.path.join(out_dir, "quality_vs_memory.png"), dpi=130, bbox_inches="tight")
    print("saved", os.path.join(out_dir, "quality_vs_memory.png"))


def evaluate_documents(model, documents, policy, budget, chunk_size, eval_start):
    results = {}
    for name, ids in documents.items():
        results[name] = run_ppl(model, ids, policy, budget, chunk_size=chunk_size, eval_start=eval_start)
    total_nll = sum(r["nll_sum_eval"] for r in results.values())
    n = sum(r["n_eval"] for r in results.values())
    return {"ppl_eval": math.exp(total_nll/n), "n_eval": n, "documents": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default="data/pride_and_prejudice.txt")
    ap.add_argument("--texts", default=None, help="comma-separated documents; overrides --text")
    ap.add_argument("--n_tokens", type=int, default=6144)
    ap.add_argument("--budgets", default="128,256,512,1024,2048")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--n_trials", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default="results/curve.json")
    args = ap.parse_args()

    model, tok, _ = build_model(args.model, args.device, args.dtype)
    paths = (args.texts or args.text).split(",")
    documents = {path: tok(load_text(path), return_tensors="pt").input_ids[:, :args.n_tokens].to(model.device) for path in paths}
    depths = [float(d) for d in args.depths.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    eval_start = max(budgets)
    if any(ids.shape[1] <= eval_start+1 for ids in documents.values()):
        raise ValueError("documents must be longer than the largest budget + 1")

    print("== full cache reference")
    full = evaluate_documents(model, documents, "full", None, args.chunk_size, eval_start)
    fn = run_niah(model, tok, "full", None, args.ctx, depths, args.n_trials, chunk_size=args.chunk_size, verbose=False)
    full.update(niah_acc=fn["accuracy"], niah=fn, kv_mb=kv_bytes(model, max(ids.shape[1] for ids in documents.values()))/2**20)
    print(f"   ppl={full['ppl_eval']:.3f}  niah={full['niah_acc']:.2f}", flush=True)

    rows = {p: [] for p in POLICY_NAMES}
    res = {"hardware": hardware_report(), "model": args.model, "dtype": str(model.dtype),
           "device": str(model.device), "model_revision": getattr(model.hf.config, "_commit_hash", None),
           "n_tokens": args.n_tokens, "ctx": args.ctx, "depths": depths, "n_trials": args.n_trials,
           "chunk_size": args.chunk_size, "eval_start": eval_start, "seed": 0,
           "document_sha256": {p: hashlib.sha256(load_text(p).encode()).hexdigest() for p in paths},
           "actual_document_tokens": {p: ids.shape[1] for p, ids in documents.items()},
           "full": full, "rows": rows, "complete": False}
    save_json(res, args.out)
    for b in budgets:
        for p in POLICY_NAMES:
            print(f"== {p} budget={b}")
            row = evaluate_documents(model, documents, p, b, args.chunk_size, eval_start)
            rn = run_niah(model, tok, p, b, args.ctx, depths, args.n_trials, chunk_size=args.chunk_size, verbose=False)
            row.update(budget=b, kv_mb=kv_bytes(model, b)/2**20, niah_acc=rn["accuracy"],
                       niah_per_depth=rn["per_depth"], niah=rn)
            rows[p].append(row)
            print(f"   ppl={row['ppl_eval']:.3f} niah={row['niah_acc']:.2f} {rn['per_depth']}", flush=True)
            save_json(res, args.out)

    res["complete"] = True
    save_json(res, args.out)
    plot(res)


if __name__ == "__main__":
    main()

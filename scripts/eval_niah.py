"""Needle-in-a-haystack (passkey) retrieval under a fixed KV budget.

A 5-digit passkey is buried at a chosen depth inside filler text; after
prefilling the whole context with eviction, the model must complete
"The secret passkey is". Perplexity barely notices a missing fact - this does.

Also the targeted RoPE test: run with --rope_mode stale vs recompute
and a needle that *survives* eviction (e.g. depth 0.95 with streaming, or any
depth with h2o). If positions are handled wrong the fact is in the cache but
cannot be read out.

    python -m scripts.eval_niah --policy streaming --budget 512 --ctx 2048 --depths 0.1,0.5,0.9,0.97
"""
from __future__ import annotations
import argparse
import random
import re
import math

import torch

from kvcache.policies import make_policy, POLICY_NAMES
from kvcache.streaming import StreamingRunner
from kvcache.utils import build_model, save_json, DEFAULT_MODEL

INTRO = ("There is important information hidden inside a lot of irrelevant text. "
         "Find it and memorize it. I will quiz you about it later.\n")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
QUESTION = "\nWhat is the secret passkey? The secret passkey is"


def build_prompt(tok, ctx_len: int, depth: float, passkey: str, device):
    if not 0 <= depth <= 1:
        raise ValueError("depth must lie in [0, 1]")
    needle = f"\nThe secret passkey is {passkey}. Remember it. {passkey} is the secret passkey.\n"
    intro = tok(INTRO, return_tensors="pt").input_ids[0]
    needle_ids = tok(needle, return_tensors="pt").input_ids[0]
    q_ids = tok(QUESTION, return_tensors="pt").input_ids[0]
    n_fill = ctx_len - len(intro) - len(needle_ids) - len(q_ids)
    if n_fill < 0:
        raise ValueError("context is shorter than the intro, needle, and question")
    reps = n_fill // len(tok(FILLER).input_ids) + 2
    fill = tok(FILLER * reps, return_tensors="pt").input_ids[0][:n_fill]
    cut = int(depth * n_fill)
    ids = torch.cat([intro, fill[:cut], needle_ids, fill[cut:], q_ids])
    needle_start = len(intro) + cut
    return ids[None].to(device), needle_start, needle_start + len(needle_ids)


def needle_retention(cache, start, end):
    """Complete needle retention, per layer/KV head, BEFORE decoding."""
    fractions = [((lc.pos >= start) & (lc.pos < end)).sum(-1).float().cpu() / (end - start)
                 for lc in cache.layers]
    values = torch.stack(fractions)
    return {"per_layer_head": values.tolist(), "mean_fraction": float(values.mean()),
            "complete_in_all_heads": bool((values == 1).all()),
            "complete_head_fraction": float((values == 1).float().mean())}


def wilson(hits, count):
    z = 1.96
    p = hits / count
    center = (p + z*z / (2*count)) / (1 + z*z/count)
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1+z*z/count)
    return [max(0, center-radius), min(1, center+radius)]


def run_niah(model, tok, policy_name, budget, ctx_len, depths, n_trials=3, rope_mode="recompute",
             chunk_size=32, seed=0, verbose=True) -> dict:
    rng = random.Random(seed)
    if not depths or n_trials < 1:
        raise ValueError("provide at least one depth and one trial")
    policy = make_policy(policy_name)
    per_depth = {}
    trials = []
    keys = [str(rng.randint(10000, 99999)) for _ in range(n_trials)]
    for depth in depths:
        hits = 0
        for t in range(n_trials):
            passkey = keys[t]  # paired examples across depths, policies, and budgets
            ids, ns, ne = build_prompt(tok, ctx_len, depth, passkey, model.device)
            runner = StreamingRunner(model, policy, budget=budget, rope_mode=rope_mode, chunk_size=chunk_size)
            logits = runner.prefill(ids)
            retained = needle_retention(runner.cache, ns, ne)
            out = tok.decode(runner.decode(logits, max_new_tokens=12))
            match = re.search(r"\d+", out)
            ok = match is not None and match.group() == passkey
            hits += ok
            # was the needle still physically in the (layer 0, head 0) cache at the end?
            in_cache = retained["complete_in_all_heads"]
            trials.append({"depth": depth, "trial": t, "passkey": passkey, "output": out,
                           "correct": ok, "needle_start": ns, "needle_end": ne,
                           "retention": retained, "actual_ctx": ids.shape[1],
                           "peak_kv_bytes": runner.peak_kv_bytes})
            if verbose:
                print(f"  ctx={ctx_len} depth={depth:.2f} trial={t} key={passkey} -> '{out.strip()[:20]}' "
                      f"{'OK' if ok else 'MISS'}  needle_in_cache={in_cache}")
        per_depth[str(depth)] = hits / n_trials
    acc = sum(per_depth.values()) / len(per_depth)
    return {"policy": policy_name, "budget": budget, "ctx_len": ctx_len, "rope_mode": rope_mode,
            "n_trials": n_trials, "per_depth": per_depth, "accuracy": acc,
            "seed": seed, "chunk_size": chunk_size, "trials": trials,
            "accuracy_ci95": wilson(sum(t["correct"] for t in trials), len(trials))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--policy", default="streaming", choices=POLICY_NAMES + ["full"])
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9,0.97")
    ap.add_argument("--n_trials", type=int, default=3)
    ap.add_argument("--rope_mode", default="recompute")
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok, _ = build_model(args.model, args.device, args.dtype)
    depths = [float(d) for d in args.depths.split(",")]
    budget = None if args.policy == "full" else args.budget
    res = run_niah(model, tok, args.policy, budget, args.ctx, depths, args.n_trials, args.rope_mode, args.chunk_size)
    print(f"\n{args.policy} budget={budget} ctx={args.ctx} rope={args.rope_mode}: accuracy={res['accuracy']:.2f}  {res['per_depth']}")
    save_json(res, args.out or f"results/niah_{args.policy}_{budget}_{args.ctx}_{args.rope_mode}.json")


if __name__ == "__main__":
    main()

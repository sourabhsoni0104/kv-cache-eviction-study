"""Instrument attention BEFORE designing eviction.

Runs one long prefill and, per layer/head, records:
  * sink_share      : fraction of each query's attention mass on the first 4 keys
                      (averaged over queries at position >= 64)
  * top_k_90        : fraction of *available* keys needed to cover 90% of the mass
                      (averaged over queries at position >= 256)  -> concentration
  * recv_by_pos     : mean attention received by each key position (avg over queries)
  * entropy         : mean row entropy (nats)
It does this on (a) real text and (b) uniformly random token ids. If the sink
share survives random tokens, early tokens absorb mass *regardless of content*.

    python -m scripts.instrument_attention --text data/pride_and_prejudice.txt --n_tokens 2048
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kvcache.utils import build_model, load_text, save_json, DEFAULT_MODEL


def analyse(model, ids, n_sink=4, q_min=64, keep_maps=(0, 5, 12, 23), map_size=256):
    s = model.spec
    L, H, N = s.n_layers, s.n_heads, ids.shape[1]
    sink_share = np.zeros((L, H))
    top90 = np.zeros((L, H))
    entropy = np.zeros((L, H))
    recv = np.zeros((L, N))
    maps = {}
    q_min = min(q_min, N - 1)
    if N < 2:
        raise ValueError("attention analysis requires at least two tokens")
    top_count = np.zeros(L)
    q_offset = 0
    causal_uniform = np.zeros(N)
    for qi in range(q_min, N):
        causal_uniform[:qi + 1] += 1 / (qi + 1) / (N - q_min)

    def cb(li, probs):
        p = probs[0]  # [H, chunk, keys]
        q_positions = torch.arange(q_offset, q_offset + p.shape[1], device=p.device)
        selected = q_positions >= q_min
        # sink share
        rows = p[:, selected, :]
        sink_share[li] += rows[:, :, :n_sink].sum((1, 2)).cpu().numpy() / (N - q_min)
        # concentration: keys needed for 90% mass, normalised by available keys
        qs = [i for i in range(p.shape[1]) if q_offset+i >= q_min and (q_offset+i-q_min) % 64 == 0]
        fr = []
        for qi in qs:
            available = q_offset + qi + 1
            row = p[:, qi, :available]
            srt = row.sort(dim=-1, descending=True).values.cumsum(-1)
            k90 = (srt < 0.9).sum(-1) + 1
            fr.append(k90.float() / available)
        if fr:
            top90[li] += torch.stack(fr).sum(0).cpu().numpy()
            top_count[li] += len(fr)
        # entropy
        ent = -(rows * (rows + 1e-12).log()).sum(-1)
        entropy[li] += ent.sum(-1).cpu().numpy() / (N - q_min)
        # mean attention received by each key position
        recv[li, :p.shape[-1]] += rows.mean(0).sum(0).cpu().numpy() / (N-q_min)
        if li in keep_maps and q_offset < map_size:
            maps.setdefault(li, np.zeros((min(N, map_size), min(N, map_size))))
            end = min(q_offset+p.shape[1], map_size)
            maps[li][q_offset:end, :min(map_size, p.shape[-1])] = p[0, :end-q_offset, :map_size].cpu().numpy()

    cache = model.new_cache("recompute")
    for q_offset in range(0, N, 128):
        model.forward(ids[:, q_offset:q_offset+128], cache, attn_callback=cb, return_last_only=True)
    top90 /= top_count[:, None]
    return dict(sink_share=sink_share, top90=top90, entropy=entropy, recv=recv, maps=maps,
                causal_uniform=causal_uniform)


def plot_all(res_text, res_rand, out_dir, n_tokens):
    os.makedirs(out_dir, exist_ok=True)
    L, H = res_text["sink_share"].shape

    # 1. sink-share heatmaps, text vs random
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, res, title in zip(axes, (res_text, res_rand), ("real text", "random tokens")):
        im = ax.imshow(res["sink_share"], aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"attention mass on first 4 tokens - {title}\n(mean over queries at pos>=64)")
        ax.set_xlabel("head"); ax.set_ylabel("layer")
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(os.path.join(out_dir, "sink_share_heatmap.png"), dpi=130, bbox_inches="tight")

    # 2. attention received by key position
    fig, ax = plt.subplots(figsize=(8, 4))
    for res, lab in ((res_text, "real text"), (res_rand, "random tokens")):
        r = res["recv"].mean(0)
        ax.plot(np.arange(1, len(r) + 1), r, label=lab, lw=1)
    ax.plot(np.arange(1, n_tokens+1), res_text["causal_uniform"], ls="--", c="gray", label="causal uniform control")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("key position (1-indexed)"); ax.set_ylabel("mean attention received")
    ax.set_title("Attention received by position (queries >=64)")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "attention_received_by_position.png"), dpi=130, bbox_inches="tight")

    # 3. concentration
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, res, title in zip(axes, (res_text, res_rand), ("real text", "random tokens")):
        im = ax.imshow(res["top90"], aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"fraction of available keys covering 90% mass - {title}")
        ax.set_xlabel("head"); ax.set_ylabel("layer")
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(os.path.join(out_dir, "attention_concentration.png"), dpi=130, bbox_inches="tight")

    # 4. example maps
    maps = res_text["maps"]
    fig, axes = plt.subplots(1, len(maps), figsize=(4 * len(maps), 4))
    for ax, (li, m) in zip(np.atleast_1d(axes), sorted(maps.items())):
        ax.imshow(np.log10(m + 1e-6), cmap="magma", vmin=-4, vmax=0)
        ax.set_title(f"layer {li}, head 0 (log10 attn)"); ax.set_xlabel("key"); ax.set_ylabel("query")
    fig.savefig(os.path.join(out_dir, "attention_maps.png"), dpi=130, bbox_inches="tight")
    print("plots written to", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default="data/pride_and_prejudice.txt")
    ap.add_argument("--n_tokens", type=int, default=2048)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    model, tok, _ = build_model(args.model, args.device, args.dtype)
    ids = tok(load_text(args.text), return_tensors="pt").input_ids[:, : args.n_tokens].to(model.device)
    torch.manual_seed(0)
    rand_ids = torch.randint(0, model.spec.vocab_size, ids.shape, device=model.device)
    # Change ALL token identities, including the initial tokens.
    shuffled_ids = ids[:, torch.randperm(ids.shape[1], device=model.device)]
    prefix_ids = ids.clone()
    prefix_ids[:, :4] = rand_ids[:, :4]

    print("analysing real text ..."); res_text = analyse(model, ids)
    print("analysing random tokens ..."); res_rand = analyse(model, rand_ids)
    print("analysing shuffled text ..."); res_shuffle = analyse(model, shuffled_ids)
    print("analysing changed prefix ..."); res_prefix = analyse(model, prefix_ids)
    plot_all(res_text, res_rand, args.out_dir, ids.shape[1])

    summary = {
        "n_tokens": int(ids.shape[1]),
        "model": args.model, "device": str(model.device), "dtype": str(model.dtype), "seed": 0,
        "sink_share_mean_shuffled": float(res_shuffle["sink_share"].mean()),
        "sink_share_mean_changed_prefix": float(res_prefix["sink_share"].mean()),
        "causal_uniform_sink_share": float(res_text["causal_uniform"][:4].sum()),
        "per_head": {name: {k: res[k].tolist() for k in ("sink_share", "top90", "entropy")}
                     for name, res in (("text", res_text), ("random", res_rand), ("shuffled", res_shuffle), ("changed_prefix", res_prefix))},
        "sink_share_mean_text": float(res_text["sink_share"].mean()),
        "sink_share_mean_random": float(res_rand["sink_share"].mean()),
        "sink_share_per_layer_text": res_text["sink_share"].mean(1).round(3).tolist(),
        "sink_share_per_layer_random": res_rand["sink_share"].mean(1).round(3).tolist(),
        "top90_fraction_mean_text": float(res_text["top90"].mean()),
        "top90_fraction_mean_random": float(res_rand["top90"].mean()),
        "entropy_mean_text": float(res_text["entropy"].mean()),
        "recv_first4_over_uniform_text": (res_text["recv"].mean(0)[:4] / res_text["causal_uniform"][:4]).round(1).tolist(),
        "recv_first4_over_uniform_random": (res_rand["recv"].mean(0)[:4] / res_rand["causal_uniform"][:4]).round(1).tolist(),
    }
    save_json(summary, "results/attention_stats.json")
    print(f"\nmean sink share: text={summary['sink_share_mean_text']:.3f}  random={summary['sink_share_mean_random']:.3f}")
    print(f"90% of mass covered by {summary['top90_fraction_mean_text']*100:.1f}% of keys (text)")
    print(f"attention on tokens 0-3 vs uniform: {summary['recv_first4_over_uniform_text']}x (text)")


if __name__ == "__main__":
    main()

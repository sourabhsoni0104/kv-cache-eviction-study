# Attention-Aware KV Cache Compression

## What this project establishes

This project compares three ways to retain a bounded transformer KV cache:
a sliding window, an explicit sink-plus-window policy, and accumulated-attention
heavy hitters. It uses Qwen2.5-0.5B-Instruct rather than random weights for the
quality experiments. The random tiny model is only a correctness fixture.

The numerical tables, hardware details, attention summaries, and plots are in
[the measured report](results/REPORT.md), generated directly from saved JSON.
Individual retrieval responses, per-head retention, document hashes, complete
loss traces, and parameters remain available for inspection.

## Key measured findings

All numbers are FP32 on Apple GPU and copied from the report.

- **Sinks are real and content-insensitive.** On natural text, 30.5% of attention
  mass lands on the first four keys, against 0.70% under causal uniform attention.
  Random IDs (29.8%) and prose with a replaced prefix (30.4%) keep the effect;
  shuffled prose lowers it to 23.0%. Token 0 receives 174× its uniform share,
  and 90% of a query's mass covers 11.1% of available keys on average.
- **Removing sinks, not a small budget, breaks the sliding window.** Full-cache
  PPL is 17.69. At B=128 (3 MiB instead of 72 MiB), sink-aware reaches 21.83 and
  heavy-hitter 21.70, while sliding reaches 157.4. Sliding at B=2048 is worst of
  all (355.0): its scored suffix begins exactly where the window first drops the
  sinks, and the loss trace shows a transient near 10³ before settling around 10².
- **Heavy-hitter gives the best perplexity, not the best retrieval.** H2O has the
  lowest PPL at every budget (18.68 vs 19.70 at B=512). Yet at B=256 it retrieves
  0/6 passkeys versus 2/6 for sink-aware, and at B=1024 2/6 versus 4/6: halving
  the recent window evicts a needle that no query has attended to yet. No
  compressed budget below 2,048 retrieved a needle at depth 0.1 under any policy.
- **Position handling must be consistent.** With identical retained needles,
  recompute and absolute reach 16.70 and 16.71 post-eviction PPL with 6/6
  retrieval; the stale negative control reaches 275.0 with 0/6.
- **Compression saves decode time as well as memory.** At 4,096 tokens, bounded
  caches decode in 0.28–0.53× the full-cache time per token. This is one
  warmed-up run; an earlier run measured 0.43–0.81×, so treat timing as noisy.

These counts come from six correlated trials per configuration; they show
direction, not precise rates.

## Model and correctness

The manual Qwen2 forward reuses the Hugging Face embeddings, attention projections,
MLP, normalization, and output weights. Its attention callback exposes each
layer's probability tensor; the custom cache exposes keys, values, original
token positions, and cumulative scores. The original HF model remains available
as `model.hf`, including its `past_key_values` interface.

The reference harness checks full and chunked logits against HF eager attention,
all three position modes before eviction, score accumulation, and policy indices.
Regression tests additionally check the bound during attention, per-layer
capacities, exact teacher-forcing alignment, invalid inputs, and an independent
RoPE re-rotation identity.

FP32 is the validated Apple GPU configuration. In an exploratory FP16 run,
HF itself differed between chunked and full prefill by up to 0.154 in logits;
matching chunk sizes reduced manual-versus-HF error below 0.05. However, the
contiguous-position-shift check still differed by 0.225, so FP16 was excluded
from the reported experiments. Numerical tolerance is not a quality guarantee.

## Instrument attention before choosing an eviction policy

For each of 24 layers and 14 query heads, the instrumentation measures:

- Attention mass landing on the first four keys, using only queries at index 64
  or later to avoid the trivial early-query effect.
- The minimum fraction of causally available keys covering 90% of a query's mass.
- Attention entropy and mean attention received by each key position.

It streams an uncompressed 2,048-token context in blocks, avoiding a full
all-layer attention-matrix allocation. Four controlled inputs are compared:
natural prose, uniformly random IDs including the prefix, shuffled prose, and
the same prose with only its first four IDs replaced.

The null baseline is **causal uniform attention**, not a flat 1/N line.
A key near the beginning is available to more queries even under uniform
attention. For a query at zero-based position q, the uniform mass per available
key is 1/(q+1). Averaging that baseline over the exact measured query set
separates learned sinks from simple exposure.

Layer/head heatmaps and numeric control results are in the report. Persistence
after prefix replacement supports content-insensitive sink behavior in these
inputs; a finite set of controls cannot establish independence for every possible
content. Global means can conceal heads that specialize in different patterns.

## Policies and a fixed budget

**Sliding window:** retain the newest available entries. It removes the initial
sinks once the window advances, and also removes older facts.

**Sink-aware, StreamingLLM-style:** protect the original first four entries and
use the remaining capacity for recent context. This stabilizes attention but
does not provide durable memory of the document's middle.

**Heavy-hitter, H2O-style:** combine recent entries with entries receiving the
largest accumulated attention mass. Query-head scores sharing a KV head are
summed. Each KV head selects its own entries in each layer; surviving indices
are sorted chronologically. Ties are deterministic and favor older entries.

The implementation reserves space for a whole incoming block before attention.
Thus B includes the new keys, and no layer's cache grows to B plus a block.
The remaining capacity B minus block length is divided according to the policy;
new entries then receive their initial scores. This is a block approximation,
not a claim of reproducing the papers' exact kernels. With block size 1 it makes
online decisions. Every policy in the measured sweep uses the same block size.

## Position handling after eviction

RoPE rotates queries and keys with their position. Its inner product depends on
the difference of those positions. Eviction alone does not necessarily invalidate
RoPE: original positions remain valid if both queries and retained keys use them
consistently.

There are two internally consistent choices:

1. **Absolute:** preserve each token's original position, including gaps.
2. **Recompute/compact:** store pre-RoPE keys and rotate retained entries at
   contiguous cache indices 0 through T-1; rotate new queries at the matching
   final indices. This is the StreamingLLM-style bounded-position convention.

Compaction changes distances across deleted gaps; it does not reproduce the
full-context model or recompute historical hidden states. It only makes the
chosen positional convention consistent. Absolute positions can eventually leave
the model's trained range; that is not established by these short experiments.

The deliberate **stale** bug caches post-RoPE keys at their insertion-time cache
positions, then reuses them after eviction while rotating new queries at compact
positions. This models naive slicing code that also resets position IDs; it is
not an assertion that every Hugging Face cache implementation has this bug.

An independent identity checks that rotating a retained key by the difference
between new and old positions equals rotating its raw key at the new position.
The learned-model ablation runs identical prompts, policy, budget, and passkeys
under all three modes. Needles are near the end and must be completely present
in **every layer and KV head before decoding**. The script fails if eviction
confounds this comparison. Full-cache retrieval is the capability control.

The report gives the actual PPL and retrieval outcomes. Fluent output alone
cannot validate RoPE; even equal retrieval accuracy on a small set does not
replace the positional identity test.

## Quality versus memory

Perplexity is evaluated on prose from *Pride and Prejudice* and *Frankenstein*,
3,072 tokens each. Every curve point, including full cache, scores the same
suffix beginning at query index 2,048, after every tested budget can evict.
Negative log-likelihoods are pooled by prediction count before exponentiation;
document perplexities are not averaged. Complete loss traces also show the
transition around first eviction.

NIAH places a five-digit passkey in exactly 2,048 prompt tokens, at depths
0.1, 0.5, and 0.9 of the filler, using two paired keys at each depth.
It greedily completes an explicit passkey question; the first returned digit
sequence must equal the key. A random answer containing the key somewhere else
does not pass. This is a completion prompt, not a chat-template benchmark.

The full-cache run measures whether this small model can solve the task.
Per-trial output and exact needle-retention fractions distinguish model
limitations, eviction, and position effects. The synthetic filler repeats,
and six paired trials per configuration are a small correlated sample; the
saved Wilson intervals are descriptive. Strong generalization claims would
require more keys, diverse fillers, and multiple models.

All policies are compared at budgets 128, 256, 512, 1,024, and 2,048.
KV bytes equal:

```text
2 (K,V) × 24 layers × 2 KV heads × 64 head dimensions × B × dtype bytes
```

In FP32 this is 24 KiB per token: 12 MiB for B=512, compared with 72 MiB for a
3,072-token full document cache. Position IDs and FP32 scores add 576 bytes per
token across layers. Model weights, tensor-copy overlap, repeated-head attention
buffers, and logits are outside this accounting. The persistent cache bound
does not imply an equally tight total-process memory bound.

The plot's full-cache PPL memory corresponds to documents, while NIAH has a
different prompt length. Decode steps also add entries to the unbounded
reference. Measured cache maxima and metadata bytes are saved alongside the
theoretical curve coordinates.

## Why sinks exist, and what to use

**Why initial tokens become sinks.** Softmax normalizes every attention row to
unit mass. A head can learn a reliable place to send attention that does not
contribute useful task information. Initial positions are visible to nearly all
later queries, making them stable candidates. Removing these learned anchors
can redistribute mass and change later activations. This is the interpretation
behind StreamingLLM, supported here by prefix and content controls; softmax
normalization alone does not mathematically require sinks.
[StreamingLLM paper](https://arxiv.org/abs/2309.17453)

**Why accumulated scores favor older entries.** The raw score is a sum over
queries since insertion. A key present for 1,000 queries has more opportunities
to accumulate mass than a key present for ten, even at the same average mass per
query. Sink behavior compounds this exposure bias. The protected recent window
gives new tokens time to collect evidence, but a fact receiving little attention
can still disappear before the eventual question reveals its importance.
Age normalization or decay reduces historical inertia but also changes sink
retention; explicit sink protection can be combined with either.
[H2O paper](https://arxiv.org/abs/2306.14048)

**Multi-turn agents.** Among these three, sink-aware streaming is a simple
baseline for stable local continuation; H2O-style retention is more suitable
when previously attended information is likely to matter again. A practical
agent should protect sinks and recent context while retaining important older
entries. Neither policy guarantees future recall. Tool results, constraints,
and decisions needing durable memory should also be stored externally and
retrieved or summarized back into context. That is an architectural recommendation,
not a workload directly evaluated by the passkey test.

**Long-document summarizers.** A sink-plus-recent cache can preserve local
perplexity while discarding the opening and middle of a document. That makes it
an unsafe default for a faithful, end-of-document summary. Prefer the full cache
when feasible, or hierarchical chunk summaries with explicit coverage and fact
retention. If forced to select one of these compressed policies, H2O is a
reasonable candidate to test because it can retain nonrecent content, but
attention popularity is not summary importance. Validate coverage and factual
recall directly; perplexity does not measure either. StreamingLLM is suitable
for producing local chunk summaries when a separate mechanism combines them.

## Limits and future work

The implementation is research code for one Qwen2 sequence, with eager attention
and a block eviction approximation. It is not an optimized serving engine.
Post-eviction hidden states still encode the context available when created.
The retrieval benchmark has repeated filler, few keys, and no multi-turn or
summarization task evaluation. Timing is one warmed-up run using a repeated
decode token, not a production throughput claim.

Per-KV-head token selection and unequal per-layer capacities are implemented and
tested. Unequal capacities between heads within a layer, learned allocation,
and a quality study of budget allocation are not implemented yet. The FP16 exploration failed a positional
numerical tolerance check and is explicitly excluded from the reported results.

## Reproduction

See [README.md](README.md). Run `bash run_all.sh` for the measured configuration.
`scripts.complete` records stage logs, `run_curve` saves completed configurations
incrementally, and `scripts.report` renders tables and diagnostic plots only
after the sweep is complete.

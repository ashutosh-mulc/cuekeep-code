# Method: pooled-KV paging

_Last updated: 2026-11-11 · Status: draft, to be verified against the original design notes_

## Notation

For one layer and one KV head, take an image region R with m tokens, keys k_1..k_m ∈ ℝ^d and
values v_1..v_m. Let q be the current decode query and τ = 1/√d.

The region's exact contribution to the softmax numerator and denominator is

    N_R = Σ_i exp(τ q·k_i) v_i          Z_R = Σ_i exp(τ q·k_i)

## Pooled stand-in with log m bias

Store k̄ = (1/m) Σ k_i and v̄ = (1/m) Σ v_i. The stand-in enters attention as a single key with an
additive logit bias b = log m:

    Ẑ_R = exp(τ q·k̄ + log m) = m · exp(τ q·k̄)          N̂_R = Ẑ_R · v̄

- **Exact in the limit:** if all k_i are equal, then Ẑ_R = Z_R and N̂_R = N_R exactly. This is
  tested in `tests/test_pooling.py`.
- **Bias direction:** by Jensen's inequality, m·exp(τ q·k̄) ≤ Σ exp(τ q·k_i). The plain stand-in
  therefore *under-weights* regions whose keys are spread along q. That is the motivation for the
  correction below.
- **When the bias matters:** it is a per-entry logit offset, so it cancels if every competing
  entry is a stand-in of the same size. It matters against text tokens, resident exact tokens and
  ragged edge regions, which is always the case in practice (notebook 2026-11-11).
- **Implementation:** in a fused attention kernel the bias is an additive mask value. For grouped-query
  attention (GQA), pool per KV head.

## Demand score (which regions to page in)

Use a second-order (log-normal) approximation of log Z_R around k̄:

    s_R(q) = τ q·k̄ + log m + ½ τ² qᵀ Σ_R q

Here Σ_R is the within-region key covariance. Store it in **low-rank** form Σ_R ≈ U_R diag(λ_R) U_Rᵀ
with rank r, so the correction costs O(r·d) per region. The score estimates the region's true
attention mass. A large gap between s_R and the plain pooled logit means the pooled stand-in is
inaccurate for this q, so fetching the region is worthwhile.

Selection rule (v0): aggregate s_R over heads (mean or max) and over the layers that share a paging
decision. Fetch the top-B regions that are not already resident. B is set by the slot pool.

> TODO(verify): the covariance correction above is our reconstruction of "query-aware, corrected by
> low-rank key covariance". Confirm it against the original design conversation.

## Memory hierarchy and eviction

| Tier | Holds | Lives on |
|---|---|---|
| Text KV | All text and generated tokens, dense | GPU |
| Pooled table | k̄, v̄, log m, U_R, λ_R for every region | GPU (small) |
| Slot pool | Exact KV for at most C regions (hard cap) | GPU |
| Host store | Exact KV for all regions | Pinned CPU memory |

- When a region is resident, mask its pooled entry and attend over its exact KV. Never count the
  region twice.
- Eviction policy (v0): evict the lowest exponential moving average (EMA) of s_R. A region fetched
  in the last `min_residency` steps cannot be evicted (hysteresis).
- Transfer: async H2D on a side stream, double-buffered (the ShadowKV pattern). A fetch decided at
  step t can be consumed at step t+1, which hides its latency.

## Knobs (all in `configs/method/*.yaml`)

`region_size g`, `cov_rank r`, `slot_capacity C`, `fetch_budget B` per step, `refresh_every k`
steps, `layers_paged` (e.g. skip the first 2 layers), `score_agg` (mean or max over heads),
`min_residency`.

## What we measure

Accuracy and trace length; ms/token; total response time; peak `torch.cuda.max_memory_allocated`;
PCIe bytes per token; hit rate (the fraction of attention mass served exactly).

# Method: pooled-KV paging

_Last updated: 2026-11-12 (after code review, see [decisions/0004](decisions/0004-review-fixes-scoring-memory-fetch.md)) · Status: draft, to be verified against the original design notes_

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
  attention (GQA), pool per KV head. Pooling and all scoring run in float32, even when the KV is bf16.

## Demand score (which regions to page in)

**Region log-mass estimate.** Let x_i = τ q·(k_i − k̄) (mean 0 over the region) and
t² = τ² qᵀ Σ_R q (the mean of x_i², with Σ_R the population covariance). Then

    log Z_R = τ q·k̄ + log m + log mean_i exp(x_i)

and the correction term log mean exp(x_i) is estimated three ways (`cov_mode`):

| Mode | Correction | Notes |
|---|---|---|
| `none` | 0 | Jensen lower bound |
| `quadratic` | ½ t² | second order (log-normal); grows without bound, so it **overshoots once logits are large** (review 2026-11-12) |
| `capped` | min(½ t², √(m−1)·t) | valid cap: a zero-mean set of m numbers with mean square t² has max ≤ √(m−1)·t, and log-mean-exp ≤ max |

Σ_R is stored in **low rank**, Σ_R ≈ B_R B_Rᵀ with B_R ∈ ℝ^{d×r}. It is computed from the eigenvectors
of the m×m Gram matrix, batched over regions of equal size, so the correction costs O(r·d) per
region at decode time.

**Default: `cov_rank: 0`** (no correction) until E001 shows the correction helps on real keys. On
synthetic keys the capped correction beats the quadratic one at large logit scales, but it does not
always beat the plain estimate.

**Per-head attention shares.** Log-masses are not comparable across heads: a head with large logits
that attends mostly to text would dominate a raw max. Each head's estimates are therefore normalised
into log *shares* of that head's attention:

    ℓ_R(q) = s_R(q) − logsumexp( other_lse(q), s_1(q), …, s_R(q) )

where `other_lse` is the log-sum-exp of the query's logits over all text and generated keys (the
attention wrapper computes it from the text KV). Selection rule (v1): aggregate ℓ_R over (head, query)
by max, or by the log of the mean share, then fetch the top-B regions not already resident.

**Open: is mass the right fetch priority?** Fetching helps most where the stand-in is wrong *and*
the region carries attention, not simply where mass is high. On synthetic keys, every score variant
ranks regions by the oracle benefit of fetching them only weakly (Spearman 0.2–0.57). E001 measures
this directly (`benefit/*`) on real keys (Q10).

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
- Eviction policy (v1): the victim is the evictable resident region with the lowest EMA of ℓ_R. A
  region is evictable once it was fetched in an earlier step and at least `min_residency` steps ago.
  A newcomer replaces the victim only if its *current* ℓ beats the victim's current ℓ by
  `admit_margin` (both sides on the same scale; review 2026-11-12).
- Host layout: exact KV is stored token-major and grouped by region (`[n, kvh, d]`, pinned), so a
  fetch is one contiguous slice: no gather, no re-pinning.
- **Fetch/commit protocol:** each fetch is an async H2D copy on a side stream that first waits for
  the compute already queued, so a reused slot is never overwritten while an earlier attention still
  reads it. A fetch stays *pending*, and attention keeps using the stand-in, until `commit` at step
  t + `lookahead`. Commit makes the compute stream wait on the fetch's CUDA event (GPU-side, no host
  sync). With `lookahead = 1`, the copy overlaps step t's compute; `lookahead = 0` is synchronous.
  An evicted region's stand-in is re-enabled immediately.

## Knobs (all in `configs/method/*.yaml`)

`region_size g`, `cov_rank r`, `cov_mode`, `slot_capacity C`, `fetch_budget B` per step,
`refresh_every k` steps, `lookahead`, `layers_paged` (e.g. skip the first 2 layers), `score_agg`
(max, or mean share, over heads and queries), `ema_decay`, `min_residency`, `admit_margin`.

## What we measure

| Metric | Definition | Why |
|---|---|---|
| Accuracy | per benchmark, mean over base seeds | Sampling decoding (model default) makes single runs noisy |
| Trace length + truncation rate | new tokens; flag when `max_new_tokens` is hit without EOS | Truncated traces censor the length distribution |
| Total response time | prefill (incl. vision encoder) + decode | Longer traces can cancel per-token savings (the ViMoD effect) |
| Prefill seconds, decode ms/token | split at the first generated token (`PhaseTracker`) | ms/token must not include prefill |
| **Visual-KV GPU bytes** | full KV: n_vis · KV bytes/token; CueKeep: pooled table + slot buffers | **The headline memory number** |
| Decode-phase peak | `max_memory_allocated` reset after the first token | Prefill holds all visual KV for every method; the whole-request peak hides the saving |
| KV breakdown | visual KV, trace KV at the end, weights | One request's visual KV (about 147 KB/token, about 150 MB per 1k tokens) is small next to the weights (about 9 GB), so savings show up as batch capacity or long-context headroom, not a single-request peak |
| PCIe bytes per token | H2D bytes moved by fetches | Fetch cost |
| Exact-mass fraction | share of attention mass served by exact KV | How well paging targets attention |

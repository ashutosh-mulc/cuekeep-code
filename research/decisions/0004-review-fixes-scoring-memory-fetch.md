# 0004: Fixes from the 2026-11-12 code review: region scoring, memory metric, fetch protocol, decoding

- **Date:** 2026-11-12
- **Status:** accepted

## Context
A review of the first implementation, against `research/method.md`, found that the stand-in math was
right, but several surrounding assumptions were not implemented as intended. Each finding was
checked numerically on CPU. Nothing here has run on a GPU yet.

## Decisions

| # | Finding | Decision |
|---|---|---|
| 1 | Raw log-masses were max-aggregated across heads, so loud heads attending to text dominated. In a constructed case, raw max found 1 of the top 8 regions; per-head shares found all 8 | Normalise per head into log attention shares, using `other_lse` from the text KV, before aggregating (`region_log_shares`) |
| 2 | The ½t² covariance correction is unbounded and overshoots at realistic logit scales. Top-8 recall fell to 0.73 vs 0.87 for plain at logit std 18 | Add `cov_mode` (none / quadratic / capped). The cap is min(½t², √(m−1)·t), a valid bound. **Default `cov_rank: 0`** until E001 shows a benefit on real keys |
| 3 | The whole-`generate` peak includes prefill, and is dominated by the weights; it cannot show the visual-KV saving | Report visual-KV GPU bytes (headline), decode-phase peak (`PhaseTracker`), and the visual / trace / weights KV breakdown. Savings are expressed as batch capacity or long-context headroom, not single-request peak |
| 4 | `sync()` before attention made every fetch synchronous; flipping the resident flags before the copy landed would let attention read half-copied slots; each fetch gathered and re-pinned | Store the host KV region-contiguous (fetch = a pinned slice). Add a pending→committed protocol with CUDA events, `lookahead` (default 1), and a copy stream that waits for queued compute before reusing a slot |
| 5 | Loaders decoded the whole dataset before subsetting; ScreenSpot-Pro list boxes were assumed xywh | Seeded subset first, decode only the visited rows. `bbox_format` / `bbox_units` must be set explicitly; errors list the actual columns |
| 6 | ms/token included prefill; truncation at `max_new_tokens` was not recorded | Prefill/decode split at the first token; NaN if no split was recorded; `truncated` flag and `truncation_rate` |
| 7 | Greedy decoding was forced | Use the model's generation_config. For Qwen3-VL-4B-Thinking (checked 2026-11-12) that is sampling with T 1.0, top_p 0.95, top_k 20, and the card's output length is 40960. Per-sample seeds; main tables average over base seeds. (The card has no explicit greedy warning; the review overstated that.) |
| 8 | E001's synthetic keys were clustered on the pooling grid and redrawn per g; logits were near-flat | Smooth spatial field shared by every g, a query-scale sweep, per-head gains. Phase A remains a code check, not a result |
| 9 | `SlotPool` compared a newcomer's raw score with the victim's EMA, and could evict a region fetched in the same step | Compare current shares on both sides (with `admit_margin`); a region fetched in the current step is never evictable |
| — | Smaller points | Multi-image `region_ids_for_spans` (`ingest` refuses a second call per layer); batched Gram-eigh covariance in place of a per-region SVD; float32 pooling and scoring; a token-order check in E000; an oracle fetch-benefit metric in E001 |

## Consequences
- E000 and E002 pre-registrations were revised before any run (noted in each README).
- Fetch priority by mass is a weak proxy for the benefit of fetching (synthetic Spearman 0.2–0.57),
  so this is now open question Q10. A better priority rule may become part of the method.
- Sampling decoding means every comparison needs several seeds (Q11). This multiplies E002 and later
  table costs by the number of seeds.

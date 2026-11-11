# Related work and novelty argument

_Last updated: 2026-11-11 · Source survey: [literature/surveys/2026-11-11-baseline-codebases-survey.md](literature/surveys/2026-11-11-baseline-codebases-survey.md)_

## Must cite and contrast head-on

| Method | Overlap with CueKeep | Where CueKeep differs | Public code | Our plan |
|---|---|---|---|---|
| **ViMoD** (arXiv 2610.12060) | Coarse group tokens plus recoverable fine groups during decoding, on Qwen3-VL-4B | ViMoD is learned (DART grouping and a TRACE router) and keeps all visual KV resident, so its peak memory is higher than full KV (42.67 vs 38.40 GiB). CueKeep is training-free and frees real memory | None | Faithful reimplementation, or match its reported Qwen3-VL-4B settings |
| **VisionPulse** (ICML 2026, arXiv 2605.31457) | Step-wise visual token selection during reasoning on Qwen3-VL-Thinking-4B | VisionPulse selects at the token level with no stand-in for dropped tokens and no offload | None | Reimplement a "per-step top-k" variant in our cache |
| **SparseVILA / -R1** (ICCV 2025) | Query-aware retrieval of visual KV at decode time | Its cache is preserved on the GPU; no pooled stand-in | None | Reimplement the retrieval rule |
| **DSTP** (arXiv 2604.12358) | Restores discarded visual tokens when attention shifts | DSTP restores tokens; CueKeep pages regions under a cap | None | Reimplement the restore rule |
| **ShadowKV** (ICML 2025) | Chunk-mean landmarks, CPU offload, sparse fetch | ShadowKV is text-only, has no pooled value, no log m bias, no region structure and no cap-driven eviction | Yes (HF 4.43) | Port its mechanism as a baseline |
| **Quest** (ICML 2024) | Query-aware page selection with min/max page keys | Quest keeps all KV on the GPU and its pages are not stand-ins | Yes (Llama) | Reimplement over regions |

## Adjacent work

- Reroute (arXiv 2606.12412): recoverable routing at prefill. Its harness is our scaffolding.
- HERMES (2601.14724): a summary token for pruned visual KV in streaming video.
- MuKV (2605.22269): multi-grained video KV with coarse-to-fine retrieval.
- HiSparse (2608.07009): a GPU hot buffer plus host-resident KV; native sparse-attention models only.
- MoBA: block-mean keys used for routing.
- Eviction baselines: SnapKV, KVzip and ExpectedAttention (via kvpress).
- Prune-once baselines: FastV, PDrop, SparseVLM, DART, DivPrune, CDPruner, VisionZip.

## Defensible novelty (the combination)

1. The pooled entry **takes part in attention** as a stand-in with a **log m bias**. It is
   training-free and exact in the limit.
2. Exact KV is **physically offloaded** to host memory, so measured peak GPU memory drops.
3. The **demand score** is query-aware and corrected by **low-rank key covariance**.
4. Eviction runs under a **hard memory cap**.
5. The evaluation targets **thinking traces of thousands of tokens** and reports total response time.

**Re-check in Jan 2027** whether ViMoD, VisionPulse or SparseVILA have released code. Log the
outcome in the notebook.

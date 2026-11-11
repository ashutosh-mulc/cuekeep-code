# 0001: Build our own HF transformers 5.x cache layer as the primary implementation

- **Date:** 2026-11-11
- **Status:** accepted

## Context
No public repo implements pooled-KV paging for visual tokens. Main results need accuracy, memory and
latency for every method, measured the same way, on Qwen3-VL-Thinking. The evidence is in
[literature/surveys/2026-11-11-baseline-codebases-survey.md](../literature/surveys/2026-11-11-baseline-codebases-survey.md).

## Options considered
1. **Fork the FastV repo**: LLaVA-only patched LLaMA forward, static prune, no Qwen-VL. Rejected (see 0002).
2. **Fork ShadowKV**: the right mechanism, but pinned to transformers 4.43 and text-only. Port the ideas instead.
3. **Vortex / SGLang sparsity**: real paged runtime, but VLM support is unverified and the Quest/HiSparse PRs are unmerged. Kept as an optional serving demo.
4. **Own `Cache` subclass in HF transformers 5.x, using the structure of the Reroute harness**: full control, and every baseline can run in the same cache.

## Decision
Option 4. `src/cuekeep/cache/paged_visual_cache.py` holds text KV, pooled region entries, a GPU slot
pool and a pinned host store. It ports ShadowKV's gather and async-H2D pattern. All baselines run
inside this cache.

## Consequences
- We write the paging logic and the kernels-in-PyTorch ourselves, so wall-clock numbers come from
  HF eager or SDPA paths rather than a serving engine.
- If reviewers question the latency numbers, revisit option 3 (M4).

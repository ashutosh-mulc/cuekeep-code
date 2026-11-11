# Overview

_Last updated: 2026-11-11_

## Problem

Thinking VLMs such as Qwen3-VL-Thinking reason for thousands of tokens about an image. Throughout
decoding, the visual tokens' KV (hundreds to thousands of entries per image, in every layer) stays
resident on the GPU and is attended at every step. Existing visual-token pruning methods fall into
two groups:

- **Prune once, before or at prefill** (FastV, PDrop, SparseVLM, DART, ...). What a long reasoning
  trace needs from the image changes over time, so evidence pruned early is lost when the trace later
  needs it. DSTP documents this "relevant visual information shift".
- **Select per step but keep everything resident** (VisionPulse, ViMoD, SparseVILA). These save
  compute, but not memory. ViMoD's reported peak memory is *higher* than full KV.

## Core idea: pooled-KV paging

1. Partition each image's merged token grid into g×g **regions**.
2. For each region, keep one **pooled entry** (mean K, mean V) on the GPU. It takes part in attention
   with a **log m bias**, where m is the region's size, so a region whose keys are all equal is
   represented *exactly*. The pooled entries act as a coarse, always-on stand-in for the whole image.
3. Move the **exact region KV** to pinned host memory.
4. At each decode step (or every k steps), compute a **query-aware demand score** for every region.
   The score is the pooled logit plus a low-rank key-covariance correction. Fetch the top regions
   into a fixed-size GPU slot pool, and **evict under a hard memory cap**.
5. When a region is resident, attention uses its exact KV in place of its pooled stand-in.

## Hypotheses

- **H1 (fidelity):** with the log m bias, the pooled stand-ins preserve attention output well enough
  that a small resident budget recovers full-KV accuracy on thinking benchmarks.
  → [E001](../experiments/E001-pooled-attention-fidelity/)
- **H2 (dynamics):** the regions a thinking trace needs shift during decoding, so recall beats
  prune-once at equal memory.
- **H3 (systems):** at a hard GPU visual-KV cap well below full KV, end-to-end latency stays
  competitive once PCIe transfer is included, and the longer traces caused by degraded perception
  are counted too.

## Claimed contributions (draft)

1. A training-free pooled stand-in with a log m bias. It is exact in the limit, and the paper should
   state an error bound in terms of within-region key spread.
2. A query-aware demand score with a low-rank covariance correction, used both to choose fetches and
   to drive eviction under a hard cap.
3. A real offload implementation that reduces measured peak GPU memory, unlike ViMoD and VisionPulse.
4. An evaluation on long thinking traces that reports total response time, PCIe bytes per token and
   peak memory alongside accuracy.

> ⚠️ This overview was reconstructed on 2026-11-11 from the baseline-codebase survey
> ([literature/surveys/2026-11-11-baseline-codebases-survey.md](literature/surveys/2026-11-11-baseline-codebases-survey.md)).
> The original design conversation was not imported. Check the details in [method.md](method.md)
> against it, especially the exact form of the demand score.

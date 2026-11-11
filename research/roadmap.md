# Roadmap

_Last updated: 2026-11-11 · Target: ICML 2027 submission (late Jan 2027; confirm the exact date when the call is out)_

Status legend: ☐ not started · ◐ in progress · ☑ done · ✗ dropped

## M0: Scaffolding (week of 2026-11-11)
- ☑ Fresh repo with the research-log and experiment structure
- ☐ Server environment set up (`scripts/setup_server.sh`); `scripts/env_check.py` passes on the GPU box
- ☐ Check Reroute's license; pin commits for Reroute, ShadowKV and kvpress in [third_party/](../third_party/README.md)

## M1: Fidelity and reference (weeks 1–2)
- ☐ **E000** environment smoke test: Qwen3-VL-4B-Thinking generates on the server with full KV; record trace lengths
- ☐ **E001** pooled-attention fidelity: dump real visual K/V and decode queries; measure stand-in error vs g, with and without the log m bias and the covariance correction
- ☐ **E002** full-KV reference: accuracy, trace length, ms/token, peak memory, on the eval suite
- ☐ Qwen3-VL adapter on transformers 5.x; `PagedVisualCache` that is correct with paging disabled (bit-match against full KV)

## M2: Paging works (weeks 3–5)
- ☐ Host store, slot pool and async H2D fetch (ShadowKV pattern)
- ☐ Demand score v0 and eviction under a cap
- ☐ Accuracy vs memory-cap curves on the thinking benchmarks

## M3: Baselines in the same cache (weeks 5–8)
- ☐ Full KV, FastV/PDrop (physical delete), Quest-over-regions, ShadowKV-style, VisionPulse-style top-k, DSTP-style restore, SnapKV (kvpress scoring), random regions
- ☐ Matched-memory and matched-latency tables

## M4: Systems numbers and ablations (weeks 8–10)
- ☐ ms/token, total response time, PCIe bytes/token, peak memory
- ☐ Ablations: log m bias, covariance rank, g, refresh interval, fetch budget, hysteresis
- ☐ Optional: Vortex or SGLang serving port (after a 1-day smoke test)

## M5: Paper (weeks 10–11)
- ☐ Draft, figures, appendix with the bound; internal review

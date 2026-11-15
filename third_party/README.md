# Third-party code: provenance and licenses

Record the **pinned commit**, the **license**, and **what we took** before copying any code. Vendored
code goes under `src/cuekeep/...`, with a header comment naming the source and the commit.
Checkouts used as references live in `third_party/src/` (gitignored).

| Project | URL | License | Pinned commit | How we use it | Status |
|---|---|---|---|---|---|
| TPRL (old `cuekeep-code` fork) | github.com/ashutosh-mulc/cuekeep-code, branch `archive/tprl` (upstream: MagicVicCoder/TPRL; Cao et al., arXiv 2603.13394) | not stated in fork; ask upstream before reuse beyond what is listed | `6617257` (2026-03-30) | Ported and rewritten: model-adapter interface, Qwen visual-span logic, MME / ScreenSpot-Pro loaders, random baseline. See research/decisions/0003 | done |
| Reroute | github.com/elmma/mllm-reroute | **unknown: check before copying** (Q6) | — | Harness structure, `physical_delete` for FastV/PDrop, KV profiler | todo |
| ShadowKV | github.com/ByteDance-Seed/ShadowKV | Apache-2.0 | — | Port: landmark selection, gather + async H2D pattern | todo |
| NVIDIA kvpress | github.com/NVIDIA/kvpress | check | — | SnapKV / KVzip scoring for eviction baselines | todo |
| Quest | github.com/mit-han-lab/Quest | check | — | Reference for min/max page scoring (reimplemented) | todo |
| VLMEvalKit | github.com/open-compass/VLMEvalKit | Apache-2.0 | — (set by setup_server.sh) | Eval harness | todo |
| Vortex | github.com/Infini-AI-Lab/vortex_torch | Apache-2.0 | — | Optional serving port (M4) | later |

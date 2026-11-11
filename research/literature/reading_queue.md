# Reading queue

Turn each item into a `papers/<citekey>.md` note once read, with frontmatter copied from the paper itself.
P0 = must contrast head-on in the paper.

| P | Paper | ID | Why | Status |
|---|---|---|---|---|
| P0 | ViMoD: Look Back, Think Ahead: Visual Memory on Demand | arXiv 2610.12060 | Closest work; Qwen3-VL-4B; memory numbers | ☐ |
| P0 | VisionPulse: Dynamic Visual Sparsity for Efficient Multimodal Reasoning | arXiv 2605.31457 | Same model family (Thinking-4B); per-step selection | ☐ |
| P0 | SparseVILA / SparseVILA-R1 | arXiv 2510.17777 | Decode-time query-aware visual KV retrieval | ☐ |
| P0 | ShadowKV | arXiv 2410.21465 | Landmarks + offload mechanism we port | ☐ |
| P0 | Quest | ICML 2024 (mit-han-lab/Quest) | Page-level query-aware selection baseline | ☐ |
| P1 | DSTP: Why and When Visual Token Pruning Fails | arXiv 2604.12358 | Evidence for H2 (attention shift) | ☐ |
| P1 | Reroute, Don't Remove | arXiv 2606.12412 | Harness + physical delete | ☐ |
| P1 | FastV | ECCV 2024 | Prune-once baseline | ☐ |
| P1 | HiSparse | arXiv 2608.07009 | Hot buffer + host KV systems design | ☐ |
| P1 | Vortex | arXiv 2606.06453 | Serving numbers; Quest slowness claim | ☐ |
| P2 | HERMES | arXiv 2601.14724 | Summary token for pruned visual KV | ☐ |
| P2 | MuKV | arXiv 2605.22269 | Multi-grained visual KV | ☐ |
| P2 | EffiVLM-Bench | arXiv 2506.00479 | Eval protocol for KV methods on VLMs | ☐ |
| P2 | KV Cache Offloading for Context-Intensive Tasks | arXiv 2604.08426 | ShadowKV on Qwen3 | ☐ |

---
date: 2026-11-11
type: survey
source: Claude design session (claude.ai/share/19bd2e91-a592-4f3a-8d70-a7dc044f7f9a); body pasted verbatim, source titles shortened
status: claims about repos are as of 2026-11; re-verify "no code found" items in Jan 2027
---

# Baseline Codebases for Pooled-KV Paging on Qwen3-VL-Thinking: Ranked Shortlist (November 2026)

Don't build on the FastV repo. No single public repo implements pooled-KV paging for visual tokens, so start from two pieces. The first is a lean Hugging Face (HF) transformers 5.x custom KV-cache layer. Use the Reroute repo (transformers 5.4, Qwen2.5-VL, physical KV deletion, lmms-eval patches, KV profiler) as scaffolding, and port in ShadowKV's offload-and-fetch design. The second is Vortex or SGLang's `mem_cache/sparsity` Quest path, which you need for credible paged-KV wall-clock numbers. The methods closest to your idea for visual tokens (ViMoD, SparseVILA, VisionPulse, DSTP) have no public code. That raises novelty risk, but it also means you will have to reimplement them as baselines.

## TL;DR
- **Primary base:** write your own `Cache` subclass for Qwen3-VL in HF transformers 5.x, starting from the Reroute repo's harness (github.com/elmma/mllm-reroute: transformers==5.4.0, Qwen2.5-VL-7B, FastV/PDrop with `physical_delete`, lmms-eval v0.7.1 patches, KV/TFLOPs profiler). Port the algorithm from ShadowKV (github.com/ByteDance-Seed/ShadowKV: chunk-mean landmarks + CPU value offload + sparse fetch). Don't use ShadowKV's code directly: it pins transformers==4.43.1 and supports text LLMs only. For serving-grade latency results, add a second implementation on Vortex (github.com/Infini-AI-Lab/vortex_torch, Apache-2.0, SGLang-based, page-centric sparse attention, Qwen3 text models) or on SGLang's in-progress Quest/HiSparse sparsity framework.
- **Baselines toolkit:** none covers FastV + SparseVLM + Quest + ShadowKV + VisionPulse on Qwen3-VL. Use lmms-eval/VLMEvalKit as the harness. Take FastV/PDrop (real deletion) from Reroute. Take DART/DivPrune/CDPruner from Visual-Token-Pruning-Benchmark or LLMC+/LightCompress. Take SnapKV/KVzip-style eviction from NVIDIA kvpress. Reimplement Quest, ShadowKV, VisionPulse and DSTP inside your own cache layer.
- **FastV repo:** no. It patches the LLaMA forward for LLaVA, and its KV-cache mode is a one-shot static prune; the FastV README describes only "a modest decrease in latency (approximately 8%)" for images, reaching "up to 25%" only for video with "around ten times image tokens". Treat FastV as an algorithm you reimplement, not a codebase.

## Key Findings

### Ranked shortlist

**1. Your own HF transformers 5.x cache layer, scaffolded from Reroute (github.com/elmma/mllm-reroute). Recommended primary base for the algorithm and the paper's main tables.**
- **Paper:** "Reroute, Don't Remove: Recoverable Visual Token Routing for VLMs" (arXiv 2606.12412).\[1\]
- **What the repo is:** a populated official release.
  - It has 22 stars and 3 commits on main.\[2\]
  - Folders: `models/`, `configs/`, `lmms_eval_patches/`, `profiler/`, `bench_data/`.\[2\]
  - Pins: torch==2.11.0+cu128, transformers==5.4.0, lmms-eval v0.7.1.\[2\]
- **What it implements:** FastV and PDrop with three routing actions (`physical_delete`, `compact_route`, `compact_route_stagewise`) across 38 configs. It ships a TFLOPs/KV-cache profiler.\[2\]
- **Models:** LLaVA-1.5-7B and Qwen2.5-VL-7B-Instruct.\[2\] There are no Qwen3-VL configs, but Qwen3-VL in transformers 5.x is structurally close.
- **License:** none was visible on the fetched page. Check before you copy code.
- **Why it ranks first:** it is the only verified public repo on a transformers 5.x stack that does real visual-KV deletion (not masking) on a Qwen-VL model, and it has an lmms-eval integration. That fits your transformers 5.20 / torch 2.8 / VLMEvalKit setup better than anything else found.
- **Coverage of your method:** low. It provides the harness, physical deletion, FastV/PDrop baselines and profiling. It has no per-region summaries, no decode-time query-aware recall, and no host offload. All of the paging logic is yours to write.

**2. ShadowKV (github.com/ByteDance-Seed/ShadowKV, ICML 2025 Spotlight). Best algorithmic and systems reference for offload-and-recall. Port its ideas; don't fork it.**
- **What it implements:**
  - Keeps a low-rank pre-RoPE key cache plus chunk landmarks and outliers on the GPU, and offloads the value cache to the CPU.\[3\]\[4\]
  - At decode time it scores chunks with the landmarks and fetches only the selected values.\[3\]
  - The ShadowKV paper (arXiv 2410.21465) says it can "reduce the GPU memory footprint of the KV cache by over 6× without accuracy degradation", supporting 6× larger batch sizes and up to 3.04× throughput on an A100 (Llama-3.1-8B) at a 1.56% sparse budget, chunk size 8, rank 160.
- **Closeness to your method:** the landmarks are means of chunks of 8 keys.\[5\] That is essentially your pooled key, without the pooled value entry or the log m bias.
- **Repo:** Apache-2.0, 316 stars. Pins transformers==4.43.1, torch==2.3.1, vllm==0.5.3.post1.\[6\]\[7\] Supported models are text LLMs only (Llama-3/3.1, GLM-4, Yi, Phi-3, Qwen2-7B; per the paper's footnote, "Needle In A Haystack is also tested on Phi-3-Mini-128K and Qwen2-7B-128K").
- **Independent reuse:** a 2026 offloading study (arXiv 2604.08426) ran experiments on Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B on top of the official ShadowKV code.\[5\] So the code can be extended to Qwen3 text models, with effort.
- **Coverage of your method:** high on mechanism (per-chunk summaries, query-aware selection, CPU offload, fetch). There is no VLM, region, or g×g structure and no explicit eviction under a cap.

**3. Vortex (github.com/Infini-AI-Lab/vortex_torch; arXiv 2606.06453, CMU/Rice/NUS). Best systems base for paged-KV wall-clock numbers.**
- **What it implements:** a Python-embedded frontend over a page-centric tensor abstraction for writing dynamic sparse attention algorithms.\[8\] The backend vendors SGLang v0.5.9.\[9\]
- **Reported results:**
  - Up to 3.46× throughput over full attention for agent-generated algorithms.\[8\]\[10\]
  - The paper says writing Double Sparsity directly in SGLang takes about 2,000 lines of code.\[8\]
  - The paper says Quest's original implementation is 44.4× slower than SGLang full attention on long-generation workloads.\[8\] That is the strongest argument against measuring wall-clock with the original Quest code.
- **Repo:** Apache-2.0, 68 stars, 258 commits, org page shows "Updated Aug 22, 2026".\[9\]\[11\]
- **Models:** README models are Qwen3-0.6B/1.7B/4B, GLM-4.7-Flash and DeepSeek-V3.\[9\] VLM support is not documented. Whether its decode hooks work with SGLang's Qwen3-VL path is unverified and is the main risk.
- **Coverage of your method:** page summaries and query-aware page selection in a real paged runtime. No documented host offload or recall.

**4. SGLang upstream `mem_cache/sparsity` + Quest PR series + HiSparse. Most complete production path, but still unfinished.**
- **What exists in `main`:**
  - A sparse-algorithm skeleton: `quest_algorithm.py`, `SparseCoordinator`, and a FlashAttention `BackendAdaptor`.\[12\]
  - SGLang's own RFC (#32657) says `create_sparse_coordinator()` "has no serving caller in `main`". In other words, it is not wired into live inference.\[12\]
- **Quest PR series (in progress, not confirmed merged):**
  - #30277 runtime integration\[13\]
  - #29666 batched page selection\[14\]
  - #31790 GPU kernels\[15\]
  - #31951 breakable CUDA Graph\[15\]
- **Experimental PR #28882:** wires Quest into the FA3 decode path for non-MLA models. On Qwen2.5-0.5B, output matched dense under greedy decoding.\[16\]
- **HiSparse:** this is exactly the "small hot GPU buffer + full KV in CPU pinned memory + on-demand swap-in kernel" design you need. However, it is documented only for models with native DeepSeek Sparse Attention (DSA), such as DeepSeek-V3.2, GLM-5.1 and DeepSeek-V4, not dense Qwen3-VL.\[17\]
- **Coverage of your method:** the eventual combination (Quest selector + HiSparse placement) is the closest systems analogue to your page fetching. Today the generic path is fragmented across open PRs.\[12\] Treat it as a target for a later serving demo, not a base for an ICML deadline.

**5. NVIDIA kvpress (github.com/NVIDIA/kvpress). Best library of eviction baselines.**
- **What it implements:** many HF-native "presses": SnapKV, ExpectedAttention, KVzip/FastKVzip/KVzap, CUR, Compactor, and `DecodingPress` for compressing during decode.\[18\]\[19\]\[20\]
- **Models tested:** LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM and Gemma3ForCausalLM.\[18\] Qwen-VL classes are not listed.
- **Releases:** PyPI lists kvpress 0.5.4, released Jul 2, 2026.
- **Coverage of your method:** eviction only, with no recall or offload. Useful for "evict-only" baselines under a matched memory cap.

### Evaluation and baseline toolkits (for recommendation b)

| Toolkit | Methods already implemented | Qwen-VL support | Real speed/memory | Notes |
|---|---|---|---|---|
| Reroute repo | FastV, PDrop (physical delete / routing) | Qwen2.5-VL-7B | Profiler (TFLOPs, KV) | transformers 5.4, lmms-eval 0.7.1\[2\] |
| Visual-Token-Pruning-Benchmark (github.com/appleiphonedddd/Visual-Token-Pruning-Benchmark) | FastV, PDrop, DART, DivPrune, CDPruner, VisionZip, PruMerge, TRIM; SparseVLM listed as 🚧 | Qwen2.5-VL-3B verified, 7B config | Prefill ms reported | Skips VisionZip on Qwen2.5-VL (no [CLS]); small community repo\[21\] |
| LLMC+ / LightCompress (github.com/ModelTC/LightCompress, AAAI 2026) | 20+ token-reduction and quantization algorithms | README lists Qwen2-VL; paper benchmarks Qwen2.5-VL | Latency/memory profiling config | Integrated with lmms-eval\[22\]\[23\]\[24\] |
| GlimpsePrune (github.com/HVision-NKU/GlimpsePrune) | Own method plus eval scripts for PDrop, VisionZip, etc. | Qwen2.5-VL-3B/7B | Prefill/decode ms | Requires training (glimpse token)\[25\]\[26\] |
| EffiVLM-Bench (paper arXiv 2506.00479) | SnapKV, PyramidKV, H2O, StreamingLLM, LOOK-M, VL-Cache + token pruning | Qwen2-VL-7B, LLaVA-OV-7B, InternVL2.5-38B\[27\] | Yes | Paper abstract states "We open-source code and recipes for EFFIVLM-BENCH" (project page effivlm-bench.github.io); repo not directly verified |
| NVIDIA kvpress | SnapKV, KVzip, DecodingPress, etc. | Qwen3 text; VLM not listed | Yes | Eviction only |
| vLLM PRs #38888 / #60963 | ViT-attention pruning; RATE image pruning | Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL | Serving | Pre-LLM pruning only, not decode-time KV; merge status unverified\[28\]\[29\] |

**Recommendation (b):** use VLMEvalKit or lmms-eval for prompts and scoring, and run every baseline through your own cache layer so that memory and ms/token are measured the same way for all of them. Source the code as follows:
- FastV/PDrop: Reroute's `physical_delete`.
- DART/DivPrune/CDPruner: Visual-Token-Pruning-Benchmark or LightCompress.
- SnapKV-style eviction: kvpress.
- Reimplement inside your cache layer:
  - Quest: min/max page keys.\[30\] Small to write; the original repo targets Llama-family text models (its README notes "[2024/10] We released Quest support for the Llama-3.1 and Mistral-v0.3 model family"), with no VLM support.
  - ShadowKV: landmarks + offload.
  - VisionPulse, DSTP and SparseVLM: no public code exists for these.

### Novelty risk: recent methods close to pooled-KV paging

| Method | What overlaps | Code |
|---|---|---|
| **ViMoD** (arXiv 2610.12060, Oct 2026) | **Closest.** Coarse group tokens linked to recoverable Fine-token groups through a Coarse-to-Fine map. Selects, retains or replaces active Fine groups during decoding. Evaluated on Qwen3-VL-4B. Learned (DART grouping + TRACE state-space router, 0.0546% extra params). Retains original visual KV banks after prefill;\[31\] peak memory 42.67 GiB vs 38.40 GiB full (per alphaxiv), so it saves compute, not memory. | **None found**\[32\]\[33\] |
| **VisionPulse** (ICML 2026, arXiv 2605.31457) | Step-wise visual token selection during reasoning on Qwen3-VL-Thinking-4B; keeps 5% of visual tokens per step and shortens traces by 11.2%. Token-level, no offload. | **None found**\[34\]\[35\]\[36\] |
| **SparseVILA** (ICCV 2025) / SparseVILA-R1 | Query-aware retrieval of visual KV during decoding from a preserved cache; supplementary material combines its strategy with Quest's generation stage. | **None found**\[37\]\[38\]\[39\] |
| **DSTP** (arXiv 2604.12358) | Restores discarded visual tokens when attention shifts during decoding.\[40\] | None (as you noted) |
| **Reroute** (arXiv 2606.12412) | Recoverable routing of excluded visual tokens across decoder stages (prefill-side). | Yes (Qwen2.5-VL)\[41\]\[42\] |
| **ShadowKV / MoBA** | Mean-pooled chunk keys as selection landmarks (text).\[30\] | ShadowKV yes |
| **HERMES** (arXiv 2601.14724) | Aggregates pruned visual KV into a single summary token (streaming video).\[43\] | Not verified |
| **MuKV** (arXiv 2605.22269) | Multi-grained (patch/frame/segment) visual KV with coarse-to-fine retrieval (video).\[44\] | Not verified |
| **HiSparse** (arXiv 2608.07009) | GPU hot buffer + host-resident full KV + swap-in (text, native sparse models). | Yes (in SGLang)\[17\]\[45\] |

**Interpretation:** "coarse summaries + on-demand recall of fine visual evidence during reasoning on Qwen3-VL-4B" is now claimed by ViMoD, and mean-pooled page keys are standard (ShadowKV, MoBA). Your defensible novelty is the combination of five things:
- The pooled entry takes part in attention as a stand-in, using the log m bias. This is a training-free, exact-in-the-limit aggregation, unlike ViMoD's learned grouping.
- Exact KV is physically offloaded to host memory, with real GPU memory savings. ViMoD increases peak memory.
- The demand score is query-aware and corrected by low-rank key covariance.
- Eviction runs under a hard memory cap.
- Evaluation targets thousands-of-token thinking traces.

ViMoD and SparseVILA must be cited and contrasted head-on. Since neither has code, compare against faithful reimplementations, or match their reported Qwen3-VL-4B settings.

## Details

### Why not the FastV repo (recommendation c)
- The official repo (github.com/pkunlp-icler/FastV, ECCV 2024 Oral) implements FastV by modifying the LLaMA forward for LLaVA HF models.\[46\]\[47\]
- Its KV-cache path prunes once at the first forward and then reuses that pruned cache. The authors note this "diverges slightly from the original version". The FastV README reports "a consistent reduction in CUDA memory usage, alongside a modest decrease in latency (approximately 8%)" for images, and says only video, "which involves around ten times image tokens", reaches "up to 25%".
- There is no Qwen-VL support, no offload, and no decode-time selection.
- Your current masking re-implementation already reproduces the algorithm. What you lack is real KV removal, and Reroute's `physical_delete` path on transformers 5.4 handles that directly.

### Suggested build plan
1. **Weeks 1–2:**
   - Fork the Reroute harness structure.
   - Add a Qwen3-VL adapter on transformers 5.x.
   - Implement a `PagedVisualCache` (a `Cache` subclass) that holds:
     - text KV (dense, GPU);
     - per-region pooled K/V plus the log m bias;
     - a host-pinned store of exact region KV;
     - a GPU slot pool for resident regions.
   - Port ShadowKV's gather and async H2D copy pattern into it.
2. **Baselines in the same cache:**
   - Full KV.
   - FastV/PDrop (physical delete).
   - Quest (min/max pages over regions).
   - ShadowKV-style (landmark + value offload).
   - VisionPulse-style per-step top-k.
   - DSTP-style restore.
   - SnapKV/KVzip via kvpress-style scoring.
3. **Measurement:**
   - Report ms/token, peak `torch.cuda.max_memory_allocated`, PCIe bytes moved per token, and total response time. Total response time matters because ViMoD shows that per-token savings can be wiped out by longer outputs.\[31\]
4. **Optional serving result:** port the region scorer to Vortex or to SGLang's Quest path once a Qwen3-VL decode path is confirmed to work there.

## Candidates Checked

| Candidate | Repo | Code status | Qwen-VL support | Framework | Real KV removal/offload |
|---|---|---|---|---|---|
| FastV | github.com/pkunlp-icler/FastV | Yes | No (LLaVA) | HF, patched LLaMA | Static prune only |
| Reroute | github.com/elmma/mllm-reroute | Yes | Qwen2.5-VL-7B | HF 5.4, lmms-eval | Physical delete\[2\] |
| ShadowKV | github.com/ByteDance-Seed/ShadowKV | Yes (old pins) | No (text LLMs) | HF 4.43.1, custom kernels | CPU value offload + fetch\[4\]\[7\] |
| Quest | github.com/mit-han-lab/Quest | Yes (dormant, 21 commits) | No (Llama-family) | HF + custom kernels | Selection only (KV kept on GPU)\[48\] |
| Vortex | github.com/Infini-AI-Lab/vortex_torch | Yes (active, Aug 2026) | Qwen3 text; VLM undocumented | SGLang v0.5.9 | Sparse page attention\[9\]\[11\] |
| SGLang Quest/HiSparse | sgl-project/sglang PRs #30277, #28882; HiSparse | Partial (PRs in progress) | HiSparse: DSA models only | SGLang | HiSparse: host offload + swap-in |
| NVIDIA kvpress | github.com/NVIDIA/kvpress | Yes (v0.5.4 on PyPI, Jul 2, 2026) | Qwen3 text; VL not listed | HF | Eviction |
| Visual-Token-Pruning-Benchmark | github.com/appleiphonedddd/Visual-Token-Pruning-Benchmark | Yes | Qwen2.5-VL-3B/7B | HF | Token pruning\[21\] |
| LLMC+/LightCompress | github.com/ModelTC/LightCompress | Yes | Qwen2-VL (README), Qwen2.5-VL (paper) | HF + lmms-eval | Token pruning\[24\] |
| GlimpsePrune | github.com/HVision-NKU/GlimpsePrune | Yes | Qwen2.5-VL-3B/7B | HF (custom) | Pruning (trained)\[25\] |
| EffiVLM-Bench | project page effivlm-bench.github.io | Claimed open; not verified | Qwen2-VL-7B | HF | KV eviction\[49\] |
| vLLM image pruning | vllm PRs #38888, #60963 | PRs (merge unverified) | Qwen2.5/3/3.5-VL | vLLM | Pre-LLM pruning\[28\]\[29\] |
| PixelPrune | github.com/OPPO-Mente-Lab/PixelPrune | Yes | Qwen3-VL, Qwen3.5-VL (vLLM 0.18 patch) | vLLM | Pre-ViT pruning\[50\] |
| SinkPruner | github.com/LaVi-Lab/SinkPruner | Stated in paper | Qwen2.5-VL | lmms-eval | Pruning\[51\]\[52\] |
| VisionPulse | none found | No | (paper: Qwen3-VL-Thinking-4B) | — | —\[53\] |
| ViMoD | none found | No | (paper: Qwen3-VL-4B) | — | —\[32\] |
| SparseVILA / -R1 | none found | No | (VILA-based) | AWQ/TinyChat | —\[54\] |
| DSTP | none | No | — | — | — |

## Caveats
- **Unverified details:**
  - Last-commit dates for Reroute, ShadowKV and Quest were not visible on the fetched pages.
  - Reroute's license was not confirmed.
  - Quest's exact transformers pin was not checked.
  - Whether the vLLM image-pruning PRs and the SGLang Quest PRs are merged was not confirmed.
- **Untested combinations:**
  - Vortex on Qwen3-VL is untested. SGLang serves VLMs, but Vortex documents only text models.
  - Run a one-day smoke test before you commit to Vortex.
- **"No code found" findings:** these for VisionPulse, ViMoD and SparseVILA come from searching papers, project pages and paper-list repos. Code could appear before ICML 2027 reviews, so recheck in January.
- **Stars, release versions and the ViMoD memory figure:** these come from repo pages and secondary summaries (alphaxiv) as of this search, not from independent runs.

## Sources

1. [\[2606.12412\] Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models](https://arxiv.org/abs/2606.12412)
2. [GitHub - elmma/mllm-reroute](https://github.com/elmma/mllm-reroute/)
3. [KV Cache Optimization Strategies for Scalable and Efficient LLM Inference](https://arxiv.org/pdf/2603.20397)
4. [ShadowKV project page](https://bytedance-seed.github.io/ShadowKV/)
5. [KV Cache Offloading for Context-Intensive Tasks](https://arxiv.org/pdf/2604.08426)
6. [GitHub - ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV)
7. [ShadowKV/requirements.txt](https://github.com/ByteDance-Seed/ShadowKV/blob/main/requirements.txt)
8. [Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents](https://arxiv.org/pdf/2606.06453)
9. [GitHub - Infini-AI-Lab/vortex_torch](https://github.com/Infini-AI-Lab/vortex_torch)
10. [Vortex project page](https://infini-ai-lab.github.io/vortex_torch/)
11. [Infini-AI-Lab · GitHub](https://github.com/Infini-AI-Lab)
12. [SGLang RFC #32657: Unified KV-Cache Sparsity Framework](https://github.com/sgl-project/sglang/issues/32657)
13. [SGLang PR #30277](https://github.com/sgl-project/sglang/pull/30277)
14. [SGLang PR #29666](https://github.com/sgl-project/sglang/pull/29666)
15. [SGLang PR #31790](https://github.com/sgl-project/sglang/pull/31790)
16. [SGLang PR #28882](https://github.com/sgl-project/sglang/pull/28882)
17. [GitHub - cloud-zby/SGLang-HiSparse](https://github.com/cloud-zby/SGLang-HiSparse)
18. [GitHub - NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
19. [kvpress releases](https://github.com/NVIDIA/kvpress/releases)
20. [kvpress fork](https://github.com/lopezalmeidaalvaro/kvpress)
21. [Visual-Token-Pruning-Benchmark](https://github.com/appleiphonedddd/Visual-Token-Pruning-Benchmark)
22. [LLMC+ (arXiv 2508.09981)](https://arxiv.org/pdf/2508.09981)
23. [LLMC+ (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/download/39598/43559)
24. [GitHub - ModelTC/LightCompress](https://github.com/ModelTC/LightCompress)
25. [GitHub - HVision-NKU/GlimpsePrune](https://github.com/HVision-NKU/GlimpsePrune)
26. [A Glimpse to Compress (arXiv 2508.01548)](https://arxiv.org/html/2508.01548v1)
27. [EffiVLM-Bench (arXiv 2506.00479)](https://arxiv.org/pdf/2506.00479)
28. [vLLM PR #38888](https://github.com/vllm-project/vllm/pull/38888)
29. [vLLM PR #60963](https://github.com/vllm-project/vllm/pull/60963)
30. [Augmenting Attention with Exponentially Decaying Memory (arXiv 2605.28640)](https://arxiv.org/pdf/2605.28640)
31. [ViMoD: Look Back, Think Ahead (arXiv 2610.12060, html)](https://arxiv.org/html/2610.12060)
32. [ViMoD (arXiv abs)](https://arxiv.org/abs/2610.12060)
33. [ViMoD (alphaxiv)](https://www.alphaxiv.org/abs/2610.12060)
34. [VisionPulse (arXiv 2605.31457)](https://arxiv.org/abs/2605.31457)
35. [Awesome-Collection-Token-Reduction](https://github.com/ZLKong/Awesome-Collection-Token-Reduction)
36. [VisionPulse (html)](https://arxiv.org/html/2605.31457)
37. [SparseVILA (arXiv 2510.17777)](https://arxiv.org/html/2510.17777)
38. [SparseVILA ICCV supplementary](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Khaki_SparseVILA_Decoupling_Visual_ICCV_2025_supplemental.pdf)
39. [SparseVILA-R1 (OpenReview)](https://openreview.net/pdf/6597aa03e44118e07d8bbbe91433da250c6beb33.pdf)
40. [DSTP: Why and When Visual Token Pruning Fails (arXiv 2604.12358)](https://arxiv.org/html/2604.12358)
41. [Reroute (HF papers)](https://huggingface.co/papers/2606.12412)
42. [Reroute (html)](https://arxiv.org/html/2606.12412)
43. [HERMES (arXiv 2601.14724)](https://arxiv.org/pdf/2601.14724)
44. [MuKV (arXiv 2605.22269)](https://arxiv.org/html/2605.22269v1)
45. [HiSparse (arXiv 2608.07009)](https://arxiv.org/html/2608.07009v1)
46. [GitHub - pkunlp-icler/FastV](https://github.com/pkunlp-icler/FastV)
47. [FastV (deepwiki)](https://deepwiki.com/pkunlp-icler/FastV)
48. [GitHub - mit-han-lab/Quest](https://github.com/mit-han-lab/Quest)
49. [EffiVLM-Bench (html)](https://arxiv.org/html/2506.00479)
50. [vLLM issue #39708: PixelPrune](https://github.com/vllm-project/vllm/issues/39708)
51. [SinkPruner (arXiv 2609.01004, html)](https://arxiv.org/html/2609.01004v1)
52. [SinkPruner (pdf)](https://arxiv.org/pdf/2609.01004)
53. [VisionPulse (pdf)](https://arxiv.org/pdf/2605.31457)
54. [SparseVILA (HF papers)](https://huggingface.co/papers/2510.17777)

"""Hookup of PagedVisualCache into HF transformers 5.x attention for Qwen3-VL.

STUB, to be written on the GPU server against the installed transformers version (roadmap M1).

Plan
----
1. Run prefill with the stock DynamicCache. Then, for each paged layer, split out the visual spans
   (from `Qwen3VLAdapter.visual_spans`), concatenate them, build ids with
   `pooling.region_ids_for_spans`, call `PagedVisualCache.ingest(layer, k_vis, v_vis, region_ids)`
   once, and **physically delete** the visual spans from the DynamicCache layer.
2. Wrap the attention of each paged decoder layer (register a custom attention function via
   `AttentionInterface`, or patch `Qwen3VLTextAttention.forward`). At decode step t it should:
     a. compute `other_lse` = logsumexp over the text/generated keys of scale·q·k, per head and query
        (needed so region scores are per-head attention *shares*);
     b. every `refresh_every` steps, call `cache.step(layer, q, t, other_lse)` (score, plan, issue async fetches);
     c. call `cache.commit(layer, t)`: fetches due by t become visible (GPU-side event wait);
     d. concatenate `[text K | cache.attention_inputs(layer)]`, build the additive float mask
        `[0... | bias]`, and run SDPA with it.
   With `lookahead=1` (default), a fetch issued in (b) at step t is committed in (c) at t+1.
3. Correctness gate: with `slot_capacity >= R` and every region resident, the outputs must
   match full KV to within bf16 tolerance (test on the server: tests/gpu/test_hf_equivalence.py).

Open points to check on the server
----------------------------------
- Positions and RoPE: HF caches *post-RoPE* keys, so the pooled keys average rotated keys (Q1 in
  research/open_questions.md). Compare with pooling pre-RoPE keys and re-rotating them at the
  region centroid's position (mrope).
- DeepStack: Qwen3-VL adds visual features to the hidden states of early LLM layers. The visual
  KV in those layers is still ordinary KV, but record which layers are skipped (`layers_paged`).
"""
from __future__ import annotations


def install_paged_attention(model, cache, cfg):  # pragma: no cover - server-only
    raise NotImplementedError("Write against the installed transformers 5.x on the GPU server; see module docstring.")

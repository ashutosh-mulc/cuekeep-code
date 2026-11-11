"""Hookup of PagedVisualCache into HF transformers 5.x attention for Qwen3-VL.

STUB, to be written on the GPU server against the installed transformers version (roadmap M1).

Plan
----
1. Run prefill with the stock DynamicCache. Then, for each paged layer, split out the visual span
   (from `Qwen3VLAdapter.visual_span`), call `PagedVisualCache.ingest(layer, k_vis, v_vis, region_ids)`,
   and **physically delete** the visual span from the DynamicCache layer.
2. Wrap the attention of each paged decoder layer (register a custom attention function via
   `AttentionInterface`, or patch `Qwen3VLTextAttention.forward`). It should:
     a. call `cache.step(layer, q, t)` every `refresh_every` steps (scoring + fetch);
     b. call `cache.sync()`;
     c. concatenate `[text K | cache.attention_inputs(layer)]` and build the additive mask
        `[0... | bias]`;
     d. run SDPA with that float mask.
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

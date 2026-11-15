"""Qwen3-VL(-Thinking) adapter for HF transformers 5.x.

Ported from TPRL's `model/qwen_mllm.py` (Qwen2.5-VL). The image-token span logic is kept, extended
with the merged grid (needed for g×g regions), and the generate() indentation bug is fixed.
Server-only: imports transformers lazily.

Decoding: by default we use the model's own generation_config. For Qwen3-VL-4B-Thinking (checked
2026-11-12) that is do_sample=True, temperature 1.0, top_p 0.95, top_k 20, and the model card
recommends out_seq_length 40960. Greedy decoding is available for debugging only. Sampling makes
every method comparison noisy, so each sample gets a fixed seed (base seed + sample index) and the
main tables average over several base seeds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..eval.metrics import PhaseTracker
from .base import VisualSpan, VLMAdapter


class Qwen3VLAdapter(VLMAdapter):
    def load(self) -> "Qwen3VLAdapter":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, self.cfg.get("dtype", "bfloat16"))
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=self.cfg.get("device_map", self.device),
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.merge = self.model.config.vision_config.spatial_merge_size
        self.image_token_id = self.model.config.image_token_id
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        return self

    # -- accounting --------------------------------------------------------------------------
    def kv_bytes_per_token(self) -> int:
        """K+V bytes per token summed over all decoder layers (2 · layers · kv_heads · head_dim · bytes)."""
        tc = self.model.config.text_config
        head_dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        elt = torch.finfo(self.model.dtype).bits // 8
        return 2 * tc.num_hidden_layers * tc.num_key_value_heads * head_dim * elt

    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    # -- inputs --------------------------------------------------------------------------------
    def build_inputs(self, image, question: str, system: Optional[str] = None) -> Dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        if system:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        return inputs.to(self.model.device)

    def visual_spans(self, inputs: Dict[str, Any]) -> List[VisualSpan]:
        ids = inputs["input_ids"][0]
        pos = torch.where(ids == self.image_token_id)[0].tolist()
        spans: List[VisualSpan] = []
        if not pos:
            return spans
        # Split the image-token positions into contiguous runs, one run per image.
        runs, start = [], pos[0]
        for a, b in zip(pos, pos[1:] + [None]):
            if b is None or b != a + 1:
                runs.append((start, a + 1))
                start = b
        grids = inputs["image_grid_thw"].tolist()
        if len(runs) != len(grids):
            raise ValueError(f"{len(runs)} image-token runs but {len(grids)} image grids")
        for (s, e), (t, h, w) in zip(runs, grids):
            gh, gw = h // self.merge, w // self.merge
            if (e - s) != t * gh * gw:
                raise ValueError(f"image span has {e - s} tokens but grid is {t}x{gh}x{gw}")
            spans.append(VisualSpan(s, e, gh, gw))
        return spans

    # -- generation ----------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], seed: Optional[int] = None, **gen_kwargs) -> Dict[str, Any]:
        max_new = int(self.cfg.get("max_new_tokens", 40960))
        gen: Dict[str, Any] = {"max_new_tokens": max_new}
        mode = self.cfg.get("decoding", "model_default")
        if mode == "greedy":
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        elif mode != "model_default":
            raise ValueError(f"decoding must be model_default or greedy, got {mode!r}")
        gen.update(gen_kwargs)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        tracker = PhaseTracker(self.model.device if torch.cuda.is_available() else None).start()
        out = self.model.generate(**inputs, streamer=tracker, **gen)
        if tracker.t_end is None:          # streamer.end() not called by this transformers version
            tracker.end()
        new = out[0, inputs["input_ids"].shape[1]:]
        n_new = int(new.numel())
        truncated = n_new >= max_new and (n_new == 0 or int(new[-1]) not in self.eos_ids)
        return {
            "text": self.processor.decode(new, skip_special_tokens=True),
            "token_ids": new.detach().cpu(),
            "num_new_tokens": n_new,
            "truncated": bool(truncated),
            "seconds": tracker.seconds,
            "prefill_seconds": tracker.prefill_seconds,
            "prefill_peak_bytes": tracker.prefill_peak,
            "decode_peak_bytes": tracker.decode_peak,
            "start_alloc_bytes": tracker.start_alloc,
        }

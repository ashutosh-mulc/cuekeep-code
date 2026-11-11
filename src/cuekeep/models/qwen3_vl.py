"""Qwen3-VL(-Thinking) adapter for HF transformers 5.x.

Ported from TPRL's `model/qwen_mllm.py` (Qwen2.5-VL). The image-token span logic is kept, extended
with the merged grid (needed for g×g regions), and the generate() indentation bug is fixed.
Server-only: imports transformers lazily.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import torch

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
        return self

    def build_inputs(self, image, question: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
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
        for (s, e), (t, h, w) in zip(runs, inputs["image_grid_thw"].tolist()):
            gh, gw = h // self.merge, w // self.merge
            assert (e - s) == t * gh * gw, f"span {e - s} != grid {t}x{gh}x{gw}"
            spans.append(VisualSpan(s, e, gh, gw))
        return spans

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], **gen_kwargs) -> Dict[str, Any]:
        gen = {"max_new_tokens": self.cfg.get("max_new_tokens", 8192), "do_sample": False, **gen_kwargs}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, **gen)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, inputs["input_ids"].shape[1]:]
        return {
            "text": self.processor.decode(new, skip_special_tokens=True),
            "num_new_tokens": int(new.numel()),
            "seconds": dt,
        }

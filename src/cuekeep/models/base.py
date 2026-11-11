"""Model adapter interface. Adapted from TPRL's `model/base_mllm.py` (see decisions/0003)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class VisualSpan:
    """Where one image's tokens sit in the input sequence, and its merged token grid."""
    start: int          # index of the first image token in input_ids
    end: int            # exclusive
    grid_h: int         # merged-token grid height (after spatial merge)
    grid_w: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


class VLMAdapter(ABC):
    def __init__(self, cfg: Dict[str, Any], device: str):
        self.cfg = cfg
        self.model_id = cfg["id"]
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> "VLMAdapter": ...

    @abstractmethod
    def build_inputs(self, image, question: str) -> Dict[str, Any]: ...

    @abstractmethod
    def visual_spans(self, inputs: Dict[str, Any]) -> List[VisualSpan]: ...

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], **gen_kwargs) -> Dict[str, Any]: ...


def get_adapter(cfg: Dict[str, Any], device: str) -> VLMAdapter:
    family = cfg.get("family", "qwen3_vl")
    if family == "qwen3_vl":
        from .qwen3_vl import Qwen3VLAdapter
        return Qwen3VLAdapter(cfg, device)
    raise ValueError(f"unknown model family: {family}")

"""Systems and accuracy metrics. Every method reports the same set (research/method.md § What we measure)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch


@dataclass
class GenerationRecord:
    sample_id: str
    correct: Optional[bool]
    num_new_tokens: int
    seconds: float
    prefill_seconds: Optional[float] = None
    peak_mem_bytes: Optional[int] = None
    h2d_bytes: int = 0
    exact_mass_frac: Optional[float] = None   # fraction of attention mass served by exact KV (paging only)

    @property
    def ms_per_token(self) -> float:
        decode = self.seconds - (self.prefill_seconds or 0.0)
        return 1000.0 * decode / max(self.num_new_tokens, 1)


@contextmanager
def cuda_peak_memory(device: Optional[int] = None) -> Iterator[Dict[str, int]]:
    """Yields a dict that holds {'peak_bytes': ...} after the block. Reports 0 on CPU."""
    out: Dict[str, int] = {"peak_bytes": 0}
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            out["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))


@contextmanager
def wall_timer() -> Iterator[Dict[str, float]]:
    out: Dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["seconds"] = time.perf_counter() - t0


@dataclass
class Summary:
    records: List[GenerationRecord] = field(default_factory=list)

    def add(self, r: GenerationRecord) -> None:
        self.records.append(r)

    def as_dict(self) -> Dict[str, float]:
        rs = self.records
        if not rs:
            return {}
        scored = [r for r in rs if r.correct is not None]
        n_tok = sum(r.num_new_tokens for r in rs)
        out = {
            "n": len(rs),
            "accuracy": (sum(r.correct for r in scored) / len(scored)) if scored else float("nan"),
            "mean_trace_tokens": n_tok / len(rs),
            "mean_response_seconds": sum(r.seconds for r in rs) / len(rs),
            "mean_ms_per_token": sum(r.ms_per_token for r in rs) / len(rs),
            "h2d_bytes_per_token": sum(r.h2d_bytes for r in rs) / max(n_tok, 1),
        }
        peaks = [r.peak_mem_bytes for r in rs if r.peak_mem_bytes]
        if peaks:
            out["max_peak_mem_gib"] = max(peaks) / 2**30
        return out

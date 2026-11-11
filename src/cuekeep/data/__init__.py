"""Datasets for quick in-repo experiments.

Main-table benchmarks run through VLMEvalKit / lmms-eval (see configs/eval/). These loaders cover
fidelity studies and smoke tests. Ported from TPRL's `data/` (see research/decisions/0003).
Every sample is normalised to: {"id", "image" (PIL), "question", "answer" | None, "bbox" | None, "meta"}.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List


def deterministic_subset(samples: List[Dict[str, Any]], n: int | None, seed: int) -> List[Dict[str, Any]]:
    """Seeded subset that does not depend on global RNG state (the old loaders used random.shuffle)."""
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    return [samples[i] for i in sorted(idx)]


def load_samples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = cfg["name"]
    if name == "mme":
        from .mme import load_mme
        rows = load_mme(cfg)
    elif name == "screenspot_pro":
        from .screenspot_pro import load_screenspot_pro
        rows = load_screenspot_pro(cfg)
    else:
        raise ValueError(f"unknown dataset: {name}")
    return deterministic_subset(rows, cfg.get("limit"), cfg.get("seed", 0))

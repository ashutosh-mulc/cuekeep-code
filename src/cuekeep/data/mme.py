"""MME (lmms-lab/MME) loader for smoke tests. Ported from TPRL's `data/mme_loader.py`."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import take_usable


def normalize_mme(s: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
    if s.get("image") is None or not s.get("question"):
        return None
    return {
        "id": str(s.get("question_id", f"mme-{i}")),
        "image": s["image"].convert("RGB"),
        "question": s["question"],
        "answer": s.get("answer"),
        "bbox": None,
        "meta": {"category": s.get("category"), "index": i},
    }


def load_mme(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_id", "lmms-lab/MME"), split=cfg.get("split", "test"))
    rows, _ = take_usable(ds, cfg.get("seed", 0), cfg.get("limit"), normalize_mme,
                      role=cfg.get("role", "all"), dev_size=cfg.get("dev_size", 0))
    return rows

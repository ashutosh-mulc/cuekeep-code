"""MME (lmms-lab/MME) loader for smoke tests. Ported from TPRL's `data/mme_loader.py`."""
from __future__ import annotations

from typing import Any, Dict, List


def load_mme(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_id", "lmms-lab/MME"), split=cfg.get("split", "test"))
    return [
        {
            "id": s.get("question_id", f"mme-{i}"),
            "image": s["image"].convert("RGB"),
            "question": s["question"],
            "answer": s.get("answer"),
            "bbox": None,
            "meta": {"category": s.get("category")},
        }
        for i, s in enumerate(ds)
    ]

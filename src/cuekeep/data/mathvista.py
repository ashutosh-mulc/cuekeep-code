"""MathVista testmini (HF `AI4Math/MathVista`, 1,000 rows): visual math reasoning with long traces.

Columns (checked 2026-11-15 on the dataset card): pid, question, image (path string),
decoded_image (image), choices, unit, precision, answer, question_type, answer_type, metadata, query.
`query` is the full prompt, including the answer-format hint, so it is used as the question.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import take_usable


def normalize_mathvista(s: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
    image = s.get("decoded_image")
    if image is None and not isinstance(s.get("image"), str):
        image = s.get("image")
    question = s.get("query") or s.get("question")
    if image is None or not question:
        return None
    return {
        "id": f"mathvista-{s.get('pid', i)}",
        "image": image.convert("RGB"),
        "question": question,
        "system": None,
        "answer": s.get("answer"),
        "bbox": None,
        "meta": {"index": i, "pid": s.get("pid"), "question_type": s.get("question_type"),
                 "answer_type": s.get("answer_type"), "choices": s.get("choices"),
                 "precision": s.get("precision"), "unit": s.get("unit")},
    }


def load_mathvista(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_id", "AI4Math/MathVista"), split=cfg.get("split", "testmini"))
    rows, skipped = take_usable(ds, cfg.get("seed", 0), cfg.get("limit"), normalize_mathvista,
                                role=cfg.get("role", "all"), dev_size=cfg.get("dev_size", 0))
    if not rows:
        raise RuntimeError(f"MathVista yielded no usable samples (skipped {skipped}); columns: {ds.column_names}")
    return rows

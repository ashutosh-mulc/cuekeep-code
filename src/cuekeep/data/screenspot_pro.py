"""ScreenSpot-Pro (high-res GUI grounding): a natural stress test for region paging.

Ported and trimmed from TPRL's `data/screen_pro_loader.py`, which was 435 lines with cache-repair and
debug printing. The schema heuristics are kept. Verify the field names once on the server and pin
them; this is logged in experiments/E000.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

_QUESTION_KEYS = ("instruction", "question", "prompt", "query", "caption")
_ANSWER_KEYS = ("answer", "response", "label", "target_description", "description")
_BBOX_KEYS = ("bbox", "target_bbox", "bbox_xywh", "bounding_box", "box", "target_box", "coordinates")

BBox = Tuple[float, float, float, float]   # x, y, w, h (in the dataset's native units)


def _text(sample: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _bbox(value: Any) -> Optional[BBox]:
    if isinstance(value, dict):
        x = value.get("x", value.get("x1", value.get("left")))
        y = value.get("y", value.get("y1", value.get("top")))
        w = value.get("w", value.get("width"))
        h = value.get("h", value.get("height"))
        if w is None and "x2" in value and x is not None:
            w = float(value["x2"]) - float(x)
        if h is None and "y2" in value and y is not None:
            h = float(value["y2"]) - float(y)
        if None not in (x, y, w, h):
            return float(x), float(y), float(w), float(h)
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return tuple(float(v) for v in value[:4])  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def load_screenspot_pro(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_id", "Voxel51/ScreenSpot-Pro"), split=cfg.get("split", "train"))
    rows, skipped = [], 0
    for i, s in enumerate(ds):
        image = s.get("image") or s.get("screenshot")
        q = _text(s, _QUESTION_KEYS)
        bb = next((b for b in (_bbox(s.get(k)) for k in _BBOX_KEYS) if b is not None), None)
        if image is None or q is None or bb is None:
            skipped += 1
            continue
        rows.append({
            "id": f"sspro-{i}", "image": image.convert("RGB"), "question": q,
            "answer": _text(s, _ANSWER_KEYS), "bbox": bb,
            "meta": {k: s[k] for k in ("application", "platform", "group", "ui_type") if k in s},
        })
    if not rows:
        raise RuntimeError(f"ScreenSpot-Pro yielded no usable samples (skipped {skipped}); check field names")
    return rows

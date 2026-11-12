"""ScreenSpot-Pro (high-res GUI grounding): a natural stress test for region paging.

Ported and trimmed from TPRL's `data/screen_pro_loader.py`, which was 435 lines with cache-repair and
debug printing.

Two things are deliberately *not* guessed (review 2026-11-12):
- **Box format.** A 4-number list can be [x, y, w, h] or [x1, y1, x2, y2], and either can be
  absolute pixels or relative [0, 1]. Set `bbox_format` (xyxy | xywh) and `bbox_units`
  (pixels | relative) in the config. A list box with no format raises an error.
- **Dataset layout.** `Voxel51/ScreenSpot-Pro` is published in FiftyOne format, and plain
  `datasets.load_dataset` may return images without annotations. If no usable sample is found, the
  error lists the columns actually present. Pin the source and field names in E000 and record them
  in its README.
Boxes are returned as [x, y, w, h] in the units given by `bbox_units` (unconverted).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import take_usable

_QUESTION_KEYS = ("instruction", "question", "prompt", "query", "caption")
_ANSWER_KEYS = ("answer", "response", "label", "target_description", "description")
_BBOX_KEYS = ("bbox", "target_bbox", "bbox_xywh", "bounding_box", "box", "target_box", "coordinates")

BBox = Tuple[float, float, float, float]   # x, y, w, h


def _text(sample: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def parse_bbox(value: Any, bbox_format: Optional[str]) -> Optional[BBox]:
    """Parse a box to (x, y, w, h). Dict boxes name their fields; list boxes need bbox_format."""
    if isinstance(value, dict):
        x = value.get("x", value.get("x1", value.get("left")))
        y = value.get("y", value.get("y1", value.get("top")))
        w = value.get("w", value.get("width"))
        h = value.get("h", value.get("height"))
        if w is None and x is not None and value.get("x2") is not None:
            w = float(value["x2"]) - float(x)
        if h is None and y is not None and value.get("y2") is not None:
            h = float(value["y2"]) - float(y)
        if None not in (x, y, w, h):
            return float(x), float(y), float(w), float(h)
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            a, b, c, d = (float(v) for v in value)
        except (TypeError, ValueError):
            return None
        if bbox_format == "xywh":
            return a, b, c, d
        if bbox_format == "xyxy":
            if c < a or d < b:
                raise ValueError(f"bbox {value} is not x1,y1,x2,y2 (x2<x1 or y2<y1); check bbox_format")
            return a, b, c - a, d - b
        raise ValueError(f"list bbox {value} is ambiguous; set bbox_format: xyxy | xywh in the dataset config")
    return None


def make_normalizer(cfg: Dict[str, Any]):
    fmt = cfg.get("bbox_format")
    bbox_keys = [cfg["bbox_field"]] if cfg.get("bbox_field") else list(_BBOX_KEYS)

    def normalize(s: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
        image = s.get("image") or s.get("screenshot")
        q = _text(s, [cfg["question_field"]] if cfg.get("question_field") else _QUESTION_KEYS)
        bb = next((b for b in (parse_bbox(s.get(k), fmt) for k in bbox_keys) if b is not None), None)
        if image is None or q is None or bb is None:
            return None
        return {
            "id": f"sspro-{i}", "image": image.convert("RGB"), "question": q,
            "answer": _text(s, _ANSWER_KEYS), "bbox": bb,
            "meta": {"index": i, "bbox_units": cfg.get("bbox_units"),
                     **{k: s[k] for k in ("application", "platform", "group", "ui_type") if k in s}},
        }

    return normalize


def load_screenspot_pro(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.get("hf_id", "Voxel51/ScreenSpot-Pro"), split=cfg.get("split", "train"))
    rows, skipped = take_usable(ds, cfg.get("seed", 0), cfg.get("limit"), make_normalizer(cfg),
                            role=cfg.get("role", "all"), dev_size=cfg.get("dev_size", 0))
    if not rows:
        raise RuntimeError(
            f"ScreenSpot-Pro yielded no usable samples (skipped {skipped}). Columns present: "
            f"{getattr(ds, 'column_names', '?')}. Set question_field / bbox_field / bbox_format in the config, "
            f"or use a source with annotations (Voxel51/ScreenSpot-Pro is FiftyOne-format)."
        )
    return rows

"""ScreenSpot-Pro (high-resolution GUI grounding): a natural stress test for region paging.

Two sources:
- `source_annotations: local` (used on the server since 2026-11-15): the original release
  (HF `likaixin/ScreenSpot-Pro`, downloaded once). `annotations_dir` holds 26 JSON files, one per
  application, each a list of entries with keys img_filename, bbox, instruction, instruction_cn, id,
  application, platform, img_size, ui_type, group. `img_filename` includes its subfolder, so the
  image is `images_dir/img_filename`. bbox is [x1, y1, x2, y2] in pixels.
  Entries are concatenated in sorted file-name order; index i = position in that list. This
  reproduces the indices of the E000 run exactly (checked 2026-11-15 against its sample ids and grids).
- otherwise: `datasets.load_dataset(hf_id)`, with explicit `question_field`, `bbox_field`,
  `bbox_format` (xyxy | xywh) and `bbox_units` (pixels | relative). A list box with no format
  raises an error, because [x, y, w, h] and [x1, y1, x2, y2] cannot be told apart.

Prompt (`prompt` in the config):
- `bare` (E000): the instruction text alone. The model then writes how-to explanations, not clicks
  (0/20 coordinate answers in E000), so this is not the benchmark task.
- `gui_click` (default from 2026-11-15): VLMEvalKit's ScreenSpot-Pro prompt, as quoted from
  vlmeval/dataset/GUI/screenspot_pro.py. A system prompt asks for `pyautogui.click(x=?, y=?)`.
  Qwen3-VL emits relative coordinates in [0, 1000] (Qwen3-VL 2d_grounding cookbook);
  `score_click` checks the point under that convention, and under absolute pixels, and records both.

Boxes are returned as (x, y, w, h) in the units given by `bbox_units` (unconverted).
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import take_usable

_QUESTION_KEYS = ("instruction", "question", "prompt", "query", "caption")
_ANSWER_KEYS = ("answer", "response", "label", "target_description", "description")
_BBOX_KEYS = ("bbox", "target_bbox", "bbox_xywh", "bounding_box", "box", "target_box", "coordinates")

BBox = Tuple[float, float, float, float]   # x, y, w, h

GUI_SYSTEM = ("You are a GUI agent. You are given a task and a screenshot of the screen. "
              "The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`")
GUI_USER = "Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"
_CLICK = re.compile(r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)")


def build_prompt(instruction: str, style: str = "gui_click") -> Tuple[Optional[str], str]:
    """(system, user) prompt for one instruction."""
    if style == "bare":
        return None, instruction
    if style == "gui_click":
        return GUI_SYSTEM, GUI_USER.format(instruction=instruction)
    raise ValueError(f"unknown ScreenSpot-Pro prompt style {style!r} (bare | gui_click)")


def score_click(text: str, bbox_xywh: BBox, img_size: Tuple[int, int]) -> Dict[str, Any]:
    """Parse the last `x=…, y=…` in the answer (after any </think>) and test it against the pixel box.

    Returns parsed (bool), the raw point, and hits under two conventions: Qwen3-VL's relative
    [0, 1000] coordinates (`hit_rel1000`) and absolute pixels (`hit_pixels`). Inclusive edges, as in VLMEvalKit.
    """
    answer = text.split("</think>")[-1]
    found = _CLICK.findall(answer) or _CLICK.findall(text)
    out: Dict[str, Any] = {"parsed": bool(found), "point": None, "hit_rel1000": False, "hit_pixels": False}
    if not found:
        return out
    x, y = (float(v) for v in found[-1])
    bx, by, bw, bh = bbox_xywh
    w, h = img_size

    def inside(px: float, py: float) -> bool:
        return bx <= px <= bx + bw and by <= py <= by + bh

    out.update(point=(x, y), hit_rel1000=inside(x / 1000 * w, y / 1000 * h), hit_pixels=inside(x, y))
    return out


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


class LocalScreenSpotPro:
    """The original release on disk, indexable like a dataset: ds[i] reads one image."""

    def __init__(self, annotations_dir: str, images_dir: str):
        files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
        if not files:
            raise FileNotFoundError(f"no *.json annotation files in {annotations_dir}")
        self.entries: List[Dict[str, Any]] = [e for f in files for e in json.load(open(f))]
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        from PIL import Image

        e = dict(self.entries[i])
        path = os.path.join(self.images_dir, e["img_filename"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"image for entry {i} ({e.get('id')}) not found: {path}")
        e["image"] = Image.open(path)
        return e


def make_normalizer(cfg: Dict[str, Any]):
    fmt = cfg.get("bbox_format")
    bbox_keys = [cfg["bbox_field"]] if cfg.get("bbox_field") else list(_BBOX_KEYS)
    q_keys = [cfg["question_field"]] if cfg.get("question_field") else list(_QUESTION_KEYS)
    style = cfg.get("prompt", "bare")

    def normalize(s: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
        image = s.get("image") or s.get("screenshot")
        instruction = _text(s, q_keys)
        bb = next((b for b in (parse_bbox(s.get(k), fmt) for k in bbox_keys) if b is not None), None)
        if image is None or instruction is None or bb is None:
            return None
        system, question = build_prompt(instruction, style)
        image = image.convert("RGB")
        return {
            "id": f"sspro-{i}", "image": image, "question": question, "system": system,
            "answer": _text(s, _ANSWER_KEYS), "bbox": bb,
            "meta": {"index": i, "instruction": instruction, "bbox_units": cfg.get("bbox_units"),
                     "img_size": tuple(s.get("img_size") or image.size), "prompt": style,
                     **{k: s[k] for k in ("id", "application", "platform", "group", "ui_type") if k in s}},
        }

    return normalize


def load_screenspot_pro(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if cfg.get("source_annotations") == "local":
        ds = LocalScreenSpotPro(cfg["annotations_dir"], cfg["images_dir"])
        cfg = {"question_field": "instruction", "bbox_field": "bbox", "bbox_format": "xyxy",
               "bbox_units": "pixels", **cfg}
    else:
        from datasets import load_dataset
        ds = load_dataset(cfg.get("hf_id", "Voxel51/ScreenSpot-Pro"), split=cfg.get("split", "train"))
    rows, skipped = take_usable(ds, cfg.get("seed", 0), cfg.get("limit"), make_normalizer(cfg),
                                role=cfg.get("role", "all"), dev_size=cfg.get("dev_size", 0))
    if not rows:
        raise RuntimeError(
            f"ScreenSpot-Pro yielded no usable samples (skipped {skipped}). Columns present: "
            f"{getattr(ds, 'column_names', '?')}. Set question_field / bbox_field / bbox_format in the config, "
            f"or use source_annotations: local with annotations_dir and images_dir."
        )
    return rows

"""Synthetic image samples for integration tests and dry runs (no download)."""
from __future__ import annotations

from typing import Any, Dict, List


def load_synthetic(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`n` images of `size` (w, h): coloured blocks, so different regions have different content."""
    from PIL import Image, ImageDraw

    n = int(cfg.get("n", 2))
    w, h = cfg.get("size", (320, 256))
    rows = []
    for i in range(n):
        img = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for j in range(6):
            x0, y0 = (37 * (i + j * 3)) % w, (53 * (i + j * 5)) % h
            d.rectangle([x0, y0, min(w - 1, x0 + w // 4), min(h - 1, y0 + h // 5)],
                        fill=((60 * j) % 256, (90 * i + 40) % 256, (30 * j + 100) % 256))
        rows.append({"id": f"synthetic-{i}", "image": img, "question": cfg.get("question", "What is shown?"),
                     "system": None, "answer": None, "bbox": None, "meta": {"index": i}})
    return rows

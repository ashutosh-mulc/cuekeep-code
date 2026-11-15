"""Rule-based answer scoring for the in-repo reference harness (E002).

Thinking models write `… </think> answer`; only the text after the last `</think>` is scored.
These rules are deliberately simple and conservative: an unparseable answer is wrong. Official
leaderboards often use an LLM judge to extract answers, so rule-based accuracy can sit a few points
below them; gate G6 allows 8 points for this and for sampling noise.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_LETTER_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-H])\)?\s*\}"),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\**\s*([A-H])\b", re.IGNORECASE),
    re.compile(r"\(([A-H])\)"),
    re.compile(r"\b([A-H])\b"),
]


def final_answer_text(text: str) -> str:
    return text.split("</think>")[-1].strip()


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, with nested braces."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{")
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            if depth == 0:
                return text[j:k].strip()
            depth -= 1
    return None


def _last_number(s: str) -> Optional[float]:
    nums = _NUM.findall(s)
    return float(nums[-1].replace(",", "")) if nums else None


def score_mathvista(text: str, answer: str, question_type: str, answer_type: str,
                    choices: Optional[List[str]] = None, precision: Optional[float] = None) -> Dict[str, Any]:
    """MathVista testmini: option letter → choice text; integer; float at `precision` decimals; list."""
    final = final_answer_text(text)
    region = last_boxed(final) or final
    pred: Any = None
    if question_type == "multi_choice" and choices:
        for pat in _LETTER_PATTERNS:
            hits = pat.findall(region) or pat.findall(final)
            hits = [h for h in hits if ord(h) - 65 < len(choices)]
            if hits:
                pred = choices[ord(hits[-1]) - 65]
                break
        if pred is None:                                  # no letter: accept an exact choice text
            present = [c for c in choices if c and c in region]
            pred = max(present, key=len) if present else None
        return {"pred": pred, "correct": pred is not None and str(pred).strip() == str(answer).strip()}
    if answer_type == "integer":
        x = _last_number(region)
        pred = None if x is None else round(x)
        return {"pred": pred, "correct": pred is not None and x == round(x) and pred == int(float(answer))}
    if answer_type == "float":
        x = _last_number(region)
        digits = int(precision) if precision is not None else 2
        pred = None if x is None else round(x, digits)
        return {"pred": pred, "correct": pred is not None and pred == round(float(answer), digits)}
    if answer_type == "list":
        m = re.findall(r"\[[^\[\]]*\]", region)
        try:
            pred = ast.literal_eval(m[-1]) if m else None
            gold = ast.literal_eval(answer)
            ok = pred is not None and [float(a) for a in pred] == [float(b) for b in gold]
        except (ValueError, SyntaxError, TypeError):
            ok = False
        return {"pred": pred, "correct": ok}
    return {"pred": region[-50:], "correct": region.strip().lower() == str(answer).strip().lower()}


def score_sample(sample: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Dispatch on the sample's dataset (from its id)."""
    sid = sample["id"]
    meta = sample.get("meta", {})
    if sid.startswith("mathvista-"):
        return score_mathvista(text, sample["answer"], meta.get("question_type"), meta.get("answer_type"),
                               meta.get("choices"), meta.get("precision"))
    if sid.startswith("sspro-"):
        from ..data.screenspot_pro import score_click
        c = score_click(text, sample["bbox"], meta["img_size"])
        return {"pred": c["point"], "correct": bool(c["hit_rel1000"]), "hit_pixels": bool(c["hit_pixels"]),
                "parsed": c["parsed"]}
    raise ValueError(f"no scorer for sample {sid}")

#!/usr/bin/env python
"""Scaffold experiments/E###-slug/ from experiments/_template and add it to the experiment index.

    python scripts/new_experiment.py "region size sweep" [--question Q1]
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
INDEX_MARKER = "<!-- index:end -->"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def next_id() -> int:
    ids = [int(m.group(1)) for p in EXP.iterdir() if (m := re.match(r"E(\d{3})-", p.name))]
    return max(ids, default=-1) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("title")
    ap.add_argument("--question", default="", help="open question this resolves, e.g. Q1")
    args = ap.parse_args()

    eid = f"E{next_id():03d}"
    folder = EXP / f"{eid}-{slugify(args.title)}"
    shutil.copytree(EXP / "_template", folder)
    today = date.today().isoformat()
    for f in folder.rglob("*"):
        if f.is_file():
            f.write_text(f.read_text().replace("{{ID}}", eid).replace("{{TITLE}}", args.title)
                         .replace("{{DATE}}", today).replace("{{QUESTION}}", args.question or "—"))

    index = EXP / "README.md"
    row = f"| [{eid}]({folder.name}/) | {args.title} | {args.question or '—'} | ☐ planned | — | {today} |\n"
    text = index.read_text()
    index.write_text(text.replace(INDEX_MARKER, row + INDEX_MARKER))
    print(f"created {folder.relative_to(ROOT)}. Write the hypothesis and success criterion before running.")


if __name__ == "__main__":
    main()

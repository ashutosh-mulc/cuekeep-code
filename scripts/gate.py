#!/usr/bin/env python
"""Show or update the gate ledger (experiments/gates.yaml).

    python scripts/gate.py list
    python scripts/gate.py set G2 passed --evidence runs/E001-pooled-attention-fidelity/20261114-101500-gate --note "criteria 1-3 met"
    python scripts/gate.py set G3 failed --evidence runs/E003-.../...-gate --note "gap 0.02 on both datasets"
    python scripts/gate.py set G5 waived --by human --note "serving path deferred; see notebook 2026-11-20"

Decide a gate only from its pre-registered thresholds (research/experiment_design.md and the
experiment's README). A failed gate is a result: record it, then follow the on-fail action.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuekeep.tracking.gates import STATUSES, load_gates, set_gate  # noqa: E402

MARK = {"pending": "☐", "passed": "✓", "failed": "✗", "waived": "~"}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("set")
    s.add_argument("gate")
    s.add_argument("status", choices=STATUSES)
    s.add_argument("--evidence", default=None)
    s.add_argument("--note", default="")
    s.add_argument("--by", default="agent", choices=("agent", "human"))
    args = ap.parse_args()

    if args.cmd == "list":
        for gid, g in load_gates().items():
            print(f"{MARK[g.get('status', 'pending')]} {gid:3s} {g.get('status', 'pending'):8s} {g['title']}")
            print(f"      experiment: {g.get('experiment') or '-'}   unlocks: {', '.join(g.get('unlocks', [])) or '-'}")
            if g.get("evidence"):
                print(f"      evidence: {g['evidence']}  ({g.get('decided')}, by {g.get('decided_by')})  {g.get('note', '')}")
        return
    try:
        g = set_gate(args.gate, args.status, args.evidence, args.note, args.by)
    except (ValueError, KeyError) as e:
        sys.exit(f"[gate] {e}")
    print(f"[gate] {args.gate} → {g['status']}")


if __name__ == "__main__":
    main()

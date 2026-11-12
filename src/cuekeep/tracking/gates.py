"""Gate ledger: small go/no-go experiments that hold back the larger ones.

`experiments/gates.yaml` lists every gate with its status. An experiment config declares
`requires_gates: [G2, ...]`, and `scripts/run.py` refuses to start it until each of those is `passed`
or `waived`. The design, thresholds and on-fail actions are in research/experiment_design.md.

Rules enforced here:
- `passed` and `failed` need evidence: a run directory, or a notebook / test reference for
  gates that are not runs.
- `waived` can only be set by the human, with a note saying why.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .config import REPO_ROOT

GATES_FILE = REPO_ROOT / "experiments" / "gates.yaml"
STATUSES = ("pending", "passed", "failed", "waived")
_HEADER = (
    "# Gate ledger. Edit with `python scripts/gate.py set ...`, not by hand.\n"
    "# Design, thresholds and on-fail actions: research/experiment_design.md\n"
)


def load_gates(path: Path = GATES_FILE) -> Dict[str, Dict[str, Any]]:
    return (yaml.safe_load(Path(path).read_text()) or {}).get("gates", {})


def check_gates(required: Iterable[str], gates: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return one message per required gate that does not allow the run to proceed (empty = clear)."""
    blocking = []
    for g in required or []:
        if g not in gates:
            blocking.append(f"{g}: not in the gate ledger")
        elif gates[g].get("status") not in ("passed", "waived"):
            blocking.append(f"{g} ({gates[g].get('title', '')}): status is {gates[g].get('status', 'pending')}")
    return blocking


def set_gate(gate_id: str, status: str, evidence: Optional[str] = None, note: str = "",
             decided_by: str = "agent", path: Path = GATES_FILE, today: Optional[str] = None) -> Dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    gates = load_gates(path)
    if gate_id not in gates:
        raise KeyError(f"unknown gate {gate_id}; known: {sorted(gates)}")
    if status in ("passed", "failed") and not evidence:
        raise ValueError(f"{status} needs --evidence (a run directory, or a test / notebook reference)")
    if status == "waived" and (decided_by != "human" or not note.strip()):
        raise ValueError("only the human can waive a gate, and a waiver needs a note saying why")
    g = gates[gate_id]
    g.update(status=status, evidence=evidence, note=note, decided_by=decided_by,
             decided=today or date.today().isoformat())
    Path(path).write_text(_HEADER + yaml.safe_dump({"gates": gates}, sort_keys=False, allow_unicode=True, width=100))
    return g

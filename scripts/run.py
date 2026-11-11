#!/usr/bin/env python
"""Run an experiment: resolve its config, open a run directory, and call the experiment's `main(cfg, run)`.

    python scripts/run.py experiments/E001-pooled-attention-fidelity/config.yaml [key=value ...] [--tag name]

The experiment ID is the config's parent folder name. The runner is `run.py` next to the config,
unless the config sets `runner:`.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuekeep.tracking import REPO_ROOT, Run, load_config  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(cfg: dict) -> str:
    want = cfg.get("device", "cuda")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if want.startswith("cuda") and not has_cuda:
        if cfg.get("allow_cpu"):
            print("[run] CUDA not available; this experiment allows CPU, continuing on cpu.")
            return "cpu"
        sys.exit("[run] CUDA not available. Run this on the GPU server (see scripts/setup_server.sh), "
                 "or set allow_cpu: true for CPU-only experiments.")
    return want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    ap.add_argument("--tag", default=None, help="suffix for the run directory name")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path, args.overrides)
    exp_dir = cfg_path.parent
    exp_id = exp_dir.name
    runner = exp_dir / cfg.get("runner", "run.py")
    if not runner.exists():
        sys.exit(f"[run] no runner at {runner}")

    cfg["device"] = resolve_device(cfg)
    seed_everything(cfg.get("seed", 0))

    spec = importlib.util.spec_from_file_location(f"{exp_id}_runner", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with Run(exp_id, cfg, tag=args.tag) as run:
        print(f"[run] {exp_id} → {run.dir.relative_to(REPO_ROOT)}")
        summary = mod.main(cfg, run) or {}
        run.finish(summary)
        print(f"[run] done: {summary}")


if __name__ == "__main__":
    main()

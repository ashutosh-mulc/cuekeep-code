"""Run directories: one folder per execution, containing everything needed to trace a number back.

runs/<exp_id>/<YYYYmmdd-HHMMSS>[-tag]/
    config.yaml      resolved config (after extends + overrides)
    meta.json        git SHA, dirty flag, command line, host, start/end time, status
    env.json         python/torch/transformers/CUDA versions, GPUs (nvidia-smi), CUDA_VISIBLE_DEVICES
    metrics.jsonl    one JSON object per log() call
    summary.json     final aggregate metrics
    log.txt          stdout/stderr from the runner (when launched via scripts/launch.sh)
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import REPO_ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def env_snapshot() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for mod in ("torch", "transformers", "accelerate", "flash_attn", "vlmeval", "lmms_eval"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env["gpus"] = [
            {"name": torch.cuda.get_device_name(i), "mem_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        pass
    try:
        env["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception:
        env["nvidia_smi"] = None
    return env


class Run:
    def __init__(self, exp_id: str, cfg: Dict[str, Any], tag: Optional[str] = None, root: Optional[Path] = None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{tag}" if tag else stamp
        self.dir = (root or REPO_ROOT / "runs") / exp_id / name
        self.dir.mkdir(parents=True, exist_ok=False)
        self.cfg = cfg
        self._t0 = time.time()
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.meta = {
            "exp_id": exp_id,
            "tag": tag,
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")) if _git("rev-parse", "HEAD") != "unknown" else None,
            "argv": sys.argv,
            "started": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }
        self._write_meta()
        (self.dir / "env.json").write_text(json.dumps(env_snapshot(), indent=2))
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._wandb = None
        if cfg.get("tracking", {}).get("wandb"):
            try:
                import wandb

                self._wandb = wandb.init(project=cfg["tracking"].get("wandb_project", "cuekeep"),
                                         name=f"{exp_id}/{name}", config=cfg, dir=str(self.dir))
            except Exception as e:  # never fail a run because of the logger
                print(f"[tracking] wandb disabled: {e}")
        if self.meta["git_dirty"]:
            print("[tracking] WARNING: uncommitted changes; results will not be reproducible from the SHA alone.")

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    def log(self, **metrics: Any) -> None:
        metrics.setdefault("t", round(time.time() - self._t0, 3))
        self._metrics.write(json.dumps(metrics, default=float) + "\n")
        self._metrics.flush()
        if self._wandb:
            self._wandb.log(metrics)

    def finish(self, summary: Dict[str, Any], status: str = "ok") -> None:
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        self.meta.update(status=status, ended=datetime.now().isoformat(timespec="seconds"),
                         seconds=round(time.time() - self._t0, 1))
        self._write_meta()
        self._metrics.close()
        if self._wandb:
            self._wandb.summary.update(summary)
            self._wandb.finish()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self.meta["status"] == "running":
            self.finish({"error": repr(exc)}, status="failed")

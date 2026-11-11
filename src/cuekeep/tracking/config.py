"""YAML configs with `extends:` inheritance and dotted CLI overrides.

    # experiments/E001-.../config.yaml
    extends: [configs/base.yaml, configs/model/qwen3_vl_4b_thinking.yaml]
    method: {region_size: 4}

Paths in `extends` are relative to the repo root. Later files win, and the file itself wins over
everything it extends. Overrides look like `method.region_size=2`, and values are parsed as YAML.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Union[str, Path], overrides: Iterable[str] = (), _seen: tuple = ()) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path) if not path.exists() else path.resolve()
    if path in _seen:
        raise ValueError(f"cyclic extends: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    parents: List[str] = raw.pop("extends", []) or []
    if isinstance(parents, str):
        parents = [parents]
    cfg: Dict[str, Any] = {}
    for p in parents:
        cfg = deep_merge(cfg, load_config(REPO_ROOT / p, (), _seen + (path,)))
    cfg = deep_merge(cfg, raw)
    for ov in overrides:
        key, _, val = ov.partition("=")
        if not _:
            raise ValueError(f"override must be key=value: {ov}")
        node = cfg
        *parts, last = key.split(".")
        for part in parts:
            node = node.setdefault(part, {})
        node[last] = yaml.safe_load(val)
    return cfg

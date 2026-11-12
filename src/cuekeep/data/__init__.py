"""Datasets for quick in-repo experiments.

Main-table benchmarks run through VLMEvalKit / lmms-eval (see configs/eval/). These loaders cover
fidelity studies and smoke tests. Ported from TPRL's `data/` (see research/decisions/0003).
Every sample is normalised to: {"id", "image" (PIL), "question", "answer" | None, "bbox" | None, "meta"}.

Rows are chosen *before* they are decoded. A seeded permutation of the dataset indices is walked,
and only the visited rows are read (`ds[i]` decodes one row), until `limit` usable samples are found.
Decoding a whole high-resolution dataset to keep 20 samples would take tens of GB of RAM.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

Row = Dict[str, Any]


def seeded_order(n: int, seed: int) -> List[int]:
    """A permutation of range(n) that depends only on (n, seed), not on global RNG state."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


DEV_SEED = 20261112   # fixes dev/test membership for every experiment; never change it once a gate has run


def split_indices(n: int, role: str = "all", dev_size: int = 0, seed: int = 0) -> List[int]:
    """Indices to walk for a split role, in a fixed order.

    all   every index, in the order given by `seed`.
    dev   the first `dev_size` indices of the DEV_SEED permutation: the subset used by gates and for
          every design or tuning choice. Smaller limits take a prefix, so gate subsets are nested.
    test  the remaining indices: never used for tuning. Main tables report on the full benchmark (to
          compare with published numbers) and on test alone (to show nothing was tuned on it).
    """
    if role == "all":
        return seeded_order(n, seed)
    if role not in ("dev", "test"):
        raise ValueError(f"role must be all, dev or test, got {role!r}")
    if not 0 < dev_size < n:
        raise ValueError(f"dev_size must be in (0, {n}) for role {role!r}, got {dev_size}")
    perm = seeded_order(n, DEV_SEED)
    return perm[:dev_size] if role == "dev" else perm[dev_size:]


def take_usable(ds, seed: int, limit: Optional[int], normalize: Callable[[Any, int], Optional[Row]],
                role: str = "all", dev_size: int = 0) -> Tuple[List[Row], int]:
    """Walk ds in a fixed order, normalising one row at a time, and stop after `limit` usable rows.

    `normalize(raw_row, index)` returns a sample dict, or None to skip. Returns (rows, n_skipped).
    `role` / `dev_size` restrict the walk to the dev or test split (see `split_indices`).
    """
    rows: List[Row] = []
    skipped = 0
    for i in split_indices(len(ds), role, dev_size, seed):
        if limit is not None and len(rows) >= limit:
            break
        r = normalize(ds[i], i)
        if r is None:
            skipped += 1
        else:
            rows.append(r)
    return rows, skipped


def load_samples(cfg: Dict[str, Any]) -> List[Row]:
    name = cfg["name"]
    if name == "mme":
        from .mme import load_mme
        return load_mme(cfg)
    if name == "screenspot_pro":
        from .screenspot_pro import load_screenspot_pro
        return load_screenspot_pro(cfg)
    raise ValueError(f"unknown dataset: {name}")

"""Random region selection: the floor every selector must beat.

Adapted from TPRL's `pruner/random_pruner.py`. It selects regions instead of tokens and takes a seeded
generator, so runs are reproducible.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def random_regions(num_regions: int, budget: int, generator: Optional[torch.Generator] = None) -> Tensor:
    """Bool mask [R] with `budget` regions marked resident."""
    budget = max(0, min(budget, num_regions))
    perm = torch.randperm(num_regions, generator=generator)
    mask = torch.zeros(num_regions, dtype=torch.bool)
    mask[perm[:budget]] = True
    return mask

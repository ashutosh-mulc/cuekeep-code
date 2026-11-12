"""E004: paging fidelity under teacher forcing (gate G4). Stub.

To be written on the server once G2 and G3 have passed (SERVER_CHECKLIST.md, Phase 4). The method,
conditions and pass criteria are fixed in this folder's README. The implementation needs the
masked-simulation attention hook, which reproduces `cuekeep.cache.attend_mixed` inside the model:
stand-ins as extra keys with a log m bias, residency through the additive mask per position.
Write the gate verdict into the returned summary as `gate/c1_equivalence`, `gate/c2_kl_vs_prune_once`,
`gate/c3_top1` and `gate/pass`, as E001 and E003 do.
"""
from __future__ import annotations


def main(cfg: dict, run) -> dict:
    raise NotImplementedError("Held until gates G2 and G3 pass; see this folder's README and SERVER_CHECKLIST.md.")

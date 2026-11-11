"""{{ID}}: {{TITLE}}. Launched by scripts/run.py, which passes the resolved config and an open Run."""
from __future__ import annotations


def main(cfg: dict, run) -> dict:
    # run.log(step=..., metric=...) writes to metrics.jsonl; return the summary dict.
    raise NotImplementedError

"""Data loading (subset before decode, explicit box formats) and metrics (phase split, truncation)."""
import math

import pytest

from cuekeep.data import seeded_order, take_usable
from cuekeep.data.screenspot_pro import parse_bbox


class CountingDataset:
    """Fake HF dataset: records which rows were read (each read = one image decode)."""

    def __init__(self, n):
        self.n, self.reads = n, []

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.reads.append(i)
        return {"i": i, "ok": i % 3 != 0}


def test_take_usable_reads_only_what_it_needs():
    ds = CountingDataset(10_000)
    rows, skipped = take_usable(ds, seed=0, limit=20, normalize=lambda s, i: s if s["ok"] else None)
    assert len(rows) == 20
    assert len(ds.reads) == 20 + skipped < 100          # never decodes the whole dataset


def test_take_usable_is_deterministic_and_seed_dependent():
    norm = lambda s, i: s                                  # noqa: E731
    a, _ = take_usable(CountingDataset(500), 1, 10, norm)
    b, _ = take_usable(CountingDataset(500), 1, 10, norm)
    c, _ = take_usable(CountingDataset(500), 2, 10, norm)
    assert [r["i"] for r in a] == [r["i"] for r in b] != [r["i"] for r in c]
    assert sorted(seeded_order(50, 0)) == list(range(50))


def test_bbox_list_requires_explicit_format():
    with pytest.raises(ValueError):
        parse_bbox([10, 20, 30, 40], None)
    assert parse_bbox([10, 20, 30, 40], "xywh") == (10, 20, 30, 40)
    assert parse_bbox([10, 20, 30, 50], "xyxy") == (10, 20, 20, 30)
    with pytest.raises(ValueError):
        parse_bbox([30, 20, 10, 40], "xyxy")               # x2 < x1: wrong format declared
    assert parse_bbox({"x1": 1, "y1": 2, "x2": 5, "y2": 9}, None) == (1, 2, 4, 7)


torch = pytest.importorskip("torch")

from cuekeep.eval import GenerationRecord, PhaseTracker, Summary  # noqa: E402


def test_phase_tracker_splits_prefill_and_decode():
    t = PhaseTracker().start()
    t.put("prompt")                                         # generate() first puts the prompt
    t.put("tok1")                                           # first generated token: prefill done
    t.put("tok2")
    t.end()
    assert t.prefill_seconds is not None and 0 <= t.prefill_seconds <= t.seconds


def test_ms_per_token_excludes_prefill_and_is_nan_without_split():
    r = GenerationRecord("a", None, num_new_tokens=11, seconds=3.0, prefill_seconds=1.0)
    assert r.ms_per_token == pytest.approx(200.0)             # 2 s over 10 decode steps
    assert math.isnan(GenerationRecord("b", None, 11, 3.0).ms_per_token)


def test_summary_reports_truncation_and_kv_breakdown():
    s = Summary()
    s.add(GenerationRecord("a", True, 100, 2.0, 0.5, truncated=False, prompt_tokens=1100, visual_tokens=1000,
                           kv_bytes_per_token=147456, visual_kv_gpu_bytes=1000 * 147456))
    s.add(GenerationRecord("b", False, 300, 4.0, 0.5, truncated=True, prompt_tokens=1100, visual_tokens=1000,
                           kv_bytes_per_token=147456))
    d = s.as_dict()
    assert d["truncation_rate"] == 0.5 and d["accuracy"] == 0.5
    assert d["mean_visual_kv_mib"] == pytest.approx(1000 * 147456 / 2**20)
    assert d["mean_visual_kv_gpu_mib"] == pytest.approx(1000 * 147456 / 2**20)

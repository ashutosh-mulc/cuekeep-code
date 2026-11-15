"""Gate ledger and enforcement, the dev/test split, the shift metrics, and the dump-based runners."""
import importlib.util
import json
import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")
torch = pytest.importorskip("torch")

from cuekeep.analysis import per_step_mass, shift_metrics, to_unit  # noqa: E402
from cuekeep.cache import attend_full, summarize_other  # noqa: E402
from cuekeep.data import DEV_SEED, seeded_order, split_indices, take_usable  # noqa: E402
from cuekeep.tracking import REPO_ROOT, Run, check_gates, load_gates, set_gate  # noqa: E402


# ----------------------------------------------------------------------------- gates

def _ledger(tmp_path):
    p = tmp_path / "gates.yaml"
    p.write_text(yaml.safe_dump({"gates": {"G1": {"title": "one", "status": "pending"},
                                           "G2": {"title": "two", "status": "pending"}}}))
    return p


def test_check_gates_blocks_until_passed_or_waived(tmp_path):
    p = _ledger(tmp_path)
    assert len(check_gates(["G1", "G2", "G9"], load_gates(p))) == 3
    set_gate("G1", "passed", evidence="runs/x", path=p)
    set_gate("G2", "waived", note="deferred", decided_by="human", path=p)
    assert check_gates(["G1", "G2"], load_gates(p)) == []
    set_gate("G1", "failed", evidence="runs/y", path=p)
    assert len(check_gates(["G1"], load_gates(p))) == 1
    assert check_gates(None, load_gates(p)) == []


def test_gate_rules(tmp_path):
    p = _ledger(tmp_path)
    with pytest.raises(ValueError):
        set_gate("G1", "passed", path=p)                                  # no evidence
    with pytest.raises(ValueError):
        set_gate("G1", "waived", note="because", decided_by="agent", path=p)   # only the human waives
    with pytest.raises(ValueError):
        set_gate("G1", "waived", decided_by="human", path=p)              # a waiver needs a reason
    with pytest.raises(KeyError):
        set_gate("G7", "passed", evidence="x", path=p)


def test_runner_refuses_a_held_experiment():
    cfg = "experiments/E001-pooled-attention-fidelity/config.main.yaml"
    assert load_gates()["G2"]["status"] != "passed", "update this test once G2 has really passed"
    r = subprocess.run([sys.executable, "scripts/run.py", cfg], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "held by gates" in (r.stderr + r.stdout) and "G2" in (r.stderr + r.stdout)


# ----------------------------------------------------------------------------- dev / test split

def test_dev_and_test_partition_and_nest():
    n = 1000
    dev, test = split_indices(n, "dev", 150), split_indices(n, "test", 150)
    assert len(dev) == 150 and len(test) == 850 and not set(dev) & set(test)
    assert sorted(dev + test) == list(range(n))
    assert dev == seeded_order(n, DEV_SEED)[:150]                          # independent of the run seed
    assert split_indices(n, "dev", 150, seed=7) == dev
    rows20, _ = take_usable(list(range(n)), 0, 20, lambda s, i: {"i": i}, role="dev", dev_size=150)
    rows8, _ = take_usable(list(range(n)), 3, 8, lambda s, i: {"i": i}, role="dev", dev_size=150)
    assert [r["i"] for r in rows8] == [r["i"] for r in rows20][:8] == dev[:8]      # gate samples are nested
    with pytest.raises(ValueError):
        split_indices(n, "dev", 0)


# ----------------------------------------------------------------------------- text summary + shift

def test_text_summary_reproduces_attention_with_text_keys():
    g = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(2, 5, 8, generator=g), torch.randn(2, 12, 8, generator=g), torch.randn(2, 12, 8, generator=g)
    tk, tv = torch.randn(2, 7, 8, generator=g), torch.randn(2, 7, 8, generator=g)
    s = 8 ** -0.5
    lse, out = summarize_other(q, tk, tv, s)
    ref = attend_full(q, torch.cat([tk, k], 1), torch.cat([tv, v], 1), s)
    torch.testing.assert_close(attend_full(q, k, v, s, lse, out), ref, rtol=1e-5, atol=1e-5)   # float32 math


def test_shift_metrics_separate_moving_from_stationary_attention():
    S, R = 40, 20
    moving = torch.full((S, R), 1e-3)
    for s in range(S):
        moving[s, (s // 4) % R] = 1.0                    # the hot region moves every 4 steps
    still = torch.full((S, R), 1e-3)
    still[:, 3] = 1.0
    m, st = shift_metrics(moving, 0.05), shift_metrics(still, 0.05)
    assert m["gap_dynamic_static"] > 0.8 and m["late_new_frac"] == 1.0 and m["turnover"] > 0.2
    assert abs(st["gap_dynamic_static"]) < 1e-6 and st["turnover"] == 0.0
    assert m["gap_dynamic_lagged"] < m["gap_dynamic_static"]              # a 1-step lag mostly keeps up


def test_per_step_mass_groups_queries_by_step():
    kvh, grp, S, R = 2, 3, 4, 5
    shares = torch.rand(kvh, grp * S, R).log()
    mass = per_step_mass(shares, S)
    ref = shares.exp().reshape(kvh, grp, S, R).sum((0, 1))
    torch.testing.assert_close(mass, ref)


# ----------------------------------------------------------------------------- runners on a fake dump

def _fake_dump(tmp_path, moving=True):
    """Two units in the dump format: 4 KV heads... kept tiny: kvh=2, heads=4, 6x8 grid, 32 steps."""
    g = torch.Generator().manual_seed(0)
    kvh, heads, d, h, w, S, nt = 2, 4, 8, 6, 8, 32, 10
    d_dir = tmp_path / "dump"
    d_dir.mkdir()
    for i, stride in enumerate((5, 1)):
        k = torch.randn(kvh, h * w, d, generator=g)
        v = torch.randn(kvh, h * w, d, generator=g)
        q = torch.randn(heads, S, d, generator=g)
        if moving:                                         # each step's queries point at a different token
            for s in range(S):
                q[:, s] = 6 * k[:, (s * 7) % (h * w)].repeat_interleave(heads // kvh, 0)
        tk, tv = torch.randn(kvh, nt, d, generator=g), torch.randn(kvh, nt, d, generator=g)
        qg = q.reshape(kvh, -1, d)
        lse, out = summarize_other(qg, tk, tv, d ** -0.5)
        attn = attend_full(qg, torch.cat([tk, k], 1), torch.cat([tv, v], 1), d ** -0.5)
        torch.save({"sample_id": f"s{i}", "dataset": "fake", "layer": 3, "grid": (h, w), "rope": "post",
                    "k": k, "v": v, "steps": torch.arange(S) * stride, "q": q,
                    "other_lse": lse.reshape(heads, S), "other_out": out.reshape(heads, S, d),
                    "attn_out": attn.reshape(heads, S, d)}, d_dir / f"unit{i}.pt")
    return d_dir


def _runner(exp):
    spec = importlib.util.spec_from_file_location(exp, REPO_ROOT / "experiments" / exp / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_to_unit_groups_dump_heads_and_reports_stride(tmp_path):
    raw = torch.load(next(iter(sorted(_fake_dump(tmp_path).glob("*.pt")))))
    u = to_unit(raw)
    assert u["q"].shape == (2, 2 * 32, 8) and u["other_lse"].shape == (2, 64) and u["steps"] == 32
    assert u["stride"] == 5 and u["name"].endswith("spread")


def test_e001_runner_on_dump_checks_capture_and_writes_verdict(tmp_path):
    from cuekeep.tracking import load_config
    cfg = load_config(REPO_ROOT / "experiments/E001-pooled-attention-fidelity/config.gate.yaml",
                      [f"dump_path={_fake_dump(tmp_path)}", "sweep.region_size=[2]", "sweep.cov_rank=[0,2]",
                       "gate.region_sizes=[2]"])
    cfg["device"] = "cpu"
    with Run("E001-test", cfg, root=tmp_path / "runs") as run:
        summary = _runner("E001-pooled-attention-fidelity").main(cfg, run)
        run.finish(summary)
    assert summary["units"] == 2
    assert summary["gate/capture_max_rel_err"] < 1e-4 and summary["gate/c1_capture"] is True
    assert isinstance(summary["gate/pass"], bool)
    assert summary["g2/budget0.25/oracle_standins"] <= summary["g2/rel_err_logm"] + 1e-6   # exact regions help
    assert json.loads((run.dir / "rows.json").read_text())


def test_e003_runner_detects_shift_on_dump(tmp_path):
    from cuekeep.tracking import load_config
    cfg = load_config(REPO_ROOT / "experiments/E003-attention-shift/config.gate.yaml",
                      [f"dump_path={_fake_dump(tmp_path, moving=True)}", "region_size=[2]", "gate.region_size=2"])
    cfg["device"] = "cpu"
    with Run("E003-test", cfg, root=tmp_path / "runs") as run:
        summary = _runner("E003-attention-shift").main(cfg, run)
        run.finish(summary)
    assert summary["units"] == 2
    assert summary["gate/gap_dynamic_static/fake"] > 0.10 and summary["gate/c1_attention_shifts"] is True
    assert summary["gate/kill"] is False and summary["gate/lag1_gap_dense"] is not None

import json

import pytest

yaml = pytest.importorskip("yaml")

from cuekeep.tracking import REPO_ROOT, Run, load_config  # noqa: E402


def test_extends_and_overrides():
    cfg = load_config(REPO_ROOT / "experiments/E001-pooled-attention-fidelity/config.yaml",
                      ["sweep.top_b=4", "model.dtype=float16"])
    assert cfg["model"]["id"] == "Qwen/Qwen3-VL-4B-Thinking"     # inherited
    assert cfg["seed"] == 0                                       # from base
    assert cfg["sweep"]["top_b"] == 4 and cfg["model"]["dtype"] == "float16"


def test_every_experiment_config_resolves_and_has_provenance():
    for cfg_path in sorted((REPO_ROOT / "experiments").glob("E[0-9][0-9][0-9]-*/config.yaml")):
        load_config(cfg_path)
        prov = yaml.safe_load((cfg_path.parent / "provenance.yaml").read_text())
        assert prov["experiment_id"] == cfg_path.parent.name.split("-")[0]
        for key in ("planned_vs_executed", "negative_results", "known_limitations"):
            assert key in prov


def test_run_dir_records_config_meta_metrics(tmp_path):
    with Run("E999-test", {"seed": 1}, tag="t", root=tmp_path) as run:
        run.log(step=0, loss=1.5)
        run.finish({"acc": 0.5})
    files = {p.name for p in run.dir.iterdir()}
    assert {"config.yaml", "meta.json", "env.json", "metrics.jsonl", "summary.json"} <= files
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "ok" and "git_sha" in meta

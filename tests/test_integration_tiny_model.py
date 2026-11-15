"""End-to-end checks of the model-facing code on a tiny random Qwen3-VL (real architecture, tokenizer and
image processor; see cuekeep.testing.tiny_qwen3vl). Runs on CPU in about a minute.

Needs transformers and the Qwen3-VL-4B-Thinking config + processor files (HF cache or Hub; no weights).
Skipped when those are unavailable. On the server, run these before any expensive job:
    pytest -q -m integration
"""
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cuekeep.tracking import REPO_ROOT  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    from cuekeep.testing.tiny_qwen3vl import build
    try:
        return build(tmp_path_factory.mktemp("tiny_qwen3vl"))
    except Exception as e:   # no network and no cached processor files
        pytest.skip(f"cannot build the tiny model: {type(e).__name__}: {e}")


@pytest.fixture(scope="session")
def adapter(tiny_dir):
    from cuekeep.models import get_adapter
    return get_adapter({"family": "qwen3_vl", "id": str(tiny_dir), "dtype": "float32", "device_map": "cpu",
                        "max_new_tokens": 32, "decoding": "greedy"}, "cpu").load()


def _sample():
    from cuekeep.data.synthetic import load_synthetic
    return load_synthetic({"n": 1, "size": (352, 288)})[0]


def test_adapter_spans_generation_and_phase_split(adapter):
    s = _sample()
    inp = adapter.build_inputs(s["image"], s["question"], system="You are a test.")
    (sp,) = adapter.visual_spans(inp)
    assert sp.num_tokens == sp.grid_h * sp.grid_w == 9 * 11                  # 288/32 x 352/32
    g = adapter.generate(inp, seed=1)
    assert g["num_new_tokens"] == g["token_ids"].numel() > 0
    assert g["prefill_seconds"] is not None and 0 < g["prefill_seconds"] <= g["seconds"]


def test_teacher_forcing_reproduces_greedy_decoding(adapter):
    from cuekeep.analysis.capture import pad_like_input_ids
    s = _sample()
    inp = adapter.build_inputs(s["image"], s["question"])
    new = adapter.generate(inp)["token_ids"]
    P, n = inp["input_ids"].shape[1], new.numel()
    kw = pad_like_input_ids(dict(inp), torch.cat([inp["input_ids"], new[None]], 1))
    with torch.inference_mode():
        logits = adapter.model(**kw, use_cache=False).logits[0]
    assert torch.equal(logits[P - 1 + torch.arange(n)].argmax(-1), new)   # same positions, same tokens


def test_capture_reconstructs_the_models_attention_output(adapter):
    from cuekeep.analysis.capture import KVCapture, pad_like_input_ids
    from cuekeep.cache import attend_full
    s = _sample()
    inp = adapter.build_inputs(s["image"], s["question"])
    (sp,) = adapter.visual_spans(inp)
    new = adapter.generate(inp)["token_ids"]
    P, n = inp["input_ids"].shape[1], new.numel()
    kw = pad_like_input_ids(dict(inp), torch.cat([inp["input_ids"], new[None]], 1))
    cap = KVCapture(adapter.model, [0, 3], (sp.start, sp.end), {"spread": P - 1 + torch.arange(n)},
                    chunk=7, store_dtype=torch.float32)
    with torch.inference_mode(), cap:
        adapter.model.model(**kw, use_cache=False)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    assert "sdpa" not in ALL_ATTENTION_FUNCTIONS._local_mapping            # hook removed on exit
    for layer in (0, 3):
        r, u = cap.results[layer], cap.results[layer]["spread"]
        H, S, d = u["q"].shape
        kvh = r["k"].shape[0]
        assert r["k"].shape == (kvh, sp.num_tokens, d) and S == n
        rec = attend_full(u["q"].reshape(kvh, -1, d), r["k"], r["v"], d ** -0.5,
                          u["other_lse"].reshape(kvh, -1), u["other_out"].reshape(kvh, -1, d))
        ref = u["attn_out"].reshape(kvh, -1, d)
        assert float(((rec - ref).norm(dim=-1) / ref.norm(dim=-1)).max()) < 1e-5


def _run(args, **env):
    r = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True,
                       env={**os.environ, **env}, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_dump_then_gate_analyses_run(tiny_dir, tmp_path):
    out = _run(["scripts/dump_visual_kv.py", "configs/dump/tiny.yaml", f"model.id={tiny_dir}", "--tag", "pytest"])
    run_dir = REPO_ROOT / out.split("[dump] → ")[1].split()[0]
    units = run_dir / "units"
    rows = [json.loads(l) for l in (run_dir / "manifest.jsonl").read_text().splitlines()]
    assert all(r["status"] == "ok" for r in rows) and len(list(units.glob("*.pt"))) == 8
    for exp in ("E003-attention-shift", "E001-pooled-attention-fidelity"):
        out = _run(["scripts/run.py", f"experiments/{exp}/config.gate.yaml", f"dump_path={units}",
                    "device=cpu", "allow_cpu=true", "--tag", "pytest", "--override-gates", "integration test"])
        rdir = REPO_ROOT / out.split(f"[run] {exp} → ")[1].split()[0]
        summary = json.loads((rdir / "summary.json").read_text())
        assert summary["units"] == 8 and isinstance(summary["gate/pass"], bool)
        if exp.startswith("E001"):
            assert summary["gate/capture_max_rel_err"] < 0.01                # bf16 storage, as on the server

"""Loaders and prompt / scoring helpers, without downloads (fake annotation files and rows)."""
import json

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from cuekeep.analysis.dumper import choose_steps  # noqa: E402
from cuekeep.data import load_samples, split_indices  # noqa: E402
from cuekeep.data.mathvista import normalize_mathvista  # noqa: E402
from cuekeep.data.screenspot_pro import GUI_USER, LocalScreenSpotPro, build_prompt, score_click  # noqa: E402

torch = pytest.importorskip("torch")


def _fake_release(tmp_path, n_per_file=(3, 2)):
    """Two annotation files in the original release format, with images at img_filename."""
    ann, img = tmp_path / "annotations", tmp_path / "images"
    ann.mkdir()
    k = 0
    for f, n in zip(("b_app_windows", "a_app_macos"), n_per_file):
        entries = []
        for j in range(n):
            rel = f"{f.split('_')[0]}_dir/shot_{k}.png"
            (img / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (400, 200), "white").save(img / rel)
            entries.append({"img_filename": rel, "bbox": [10 + k, 20, 50 + k, 60], "instruction": f"click thing {k}",
                            "instruction_cn": "", "id": f"{f}_{j}", "application": f.split("_")[0], "platform": "x",
                            "img_size": [400, 200], "ui_type": "icon", "group": "Dev"})
            k += 1
        (ann / f"{f}.json").write_text(json.dumps(entries))
    return ann, img


def test_local_screenspot_pro_order_box_and_prompt(tmp_path):
    ann, img = _fake_release(tmp_path)
    ds = LocalScreenSpotPro(str(ann), str(img))
    assert len(ds) == 5
    assert ds.entries[0]["id"] == "a_app_macos_0"                        # files in sorted name order
    rows = load_samples({"name": "screenspot_pro", "source_annotations": "local", "annotations_dir": str(ann),
                         "images_dir": str(img), "prompt": "gui_click", "role": "dev", "dev_size": 3, "limit": 2})
    dev = split_indices(5, "dev", 3)
    assert [r["meta"]["index"] for r in rows] == dev[:2]
    r = rows[0]
    e = ds.entries[dev[0]]
    x1, y1, x2, y2 = e["bbox"]
    assert r["bbox"] == (x1, y1, x2 - x1, y2 - y1)                       # xyxy pixels -> xywh
    assert r["question"] == GUI_USER.format(instruction=e["instruction"]) and "pyautogui" in r["system"]
    assert r["meta"]["img_size"] == (400, 200)


def test_screenspot_prompt_styles():
    assert build_prompt("open file", "bare") == (None, "open file")
    sys_p, user = build_prompt("open file", "gui_click")
    assert "pyautogui.click(x=?, y=?)" in sys_p and user.endswith("open file")
    with pytest.raises(ValueError):
        build_prompt("x", "nope")


def test_score_click_conventions():
    box, size = (100, 50, 40, 20), (1000, 500)                          # x 100-140, y 50-70
    rel = score_click("<think>x=1, y=1</think>pyautogui.click(x=120, y=120)", box, size)   # 0-1000 -> (120, 60)
    assert rel["parsed"] and rel["point"] == (120.0, 120.0) and rel["hit_rel1000"] and not rel["hit_pixels"]
    px = score_click("pyautogui.click(x=130, y=60)", box, size)
    assert px["hit_pixels"] and not px["hit_rel1000"]
    assert score_click("no click here", box, size)["parsed"] is False


def test_mathvista_row_uses_query_and_decoded_image():
    row = {"pid": "7", "question": "Q?", "query": "Hint: ... Question: Q?", "image": "images/7.jpg",
           "decoded_image": Image.new("RGBA", (30, 20)), "answer": "3", "question_type": "free_form",
           "answer_type": "integer", "choices": None, "precision": None, "unit": None}
    s = normalize_mathvista(row, 0)
    assert s["id"] == "mathvista-7" and s["question"].startswith("Hint") and s["image"].mode == "RGB"
    assert normalize_mathvista({**row, "decoded_image": None}, 0) is None


def test_choose_steps():
    st = choose_steps(1000, 64, dense_len=256, dense_start_frac=0.25)
    assert st["spread"].numel() == 64 and st["spread"][0] == 0 and st["spread"][-1] == 999
    assert st["dense"].tolist() == list(range(250, 506))
    short = choose_steps(40, 64, dense_len=256)
    assert short["spread"].tolist() == list(range(40)) and "dense" not in short    # window < 32 steps
    assert "dense" not in choose_steps(1000, 64)                                    # dense not requested

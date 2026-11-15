"""Rule-based answer scoring (E002)."""
from cuekeep.eval.scoring import final_answer_text, last_boxed, score_mathvista, score_sample

CH = ["135°", "140°", "145°", "150°"]


def test_final_answer_and_boxed():
    assert final_answer_text("<think>A is wrong</think>\n The answer is C") == "The answer is C"
    assert last_boxed(r"x \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed("no box") is None


def test_multi_choice_letter_forms():
    for text in ("</think>The answer is C.", "</think>(C) 145°", r"</think>\boxed{C}", "</think>Answer: **C**"):
        assert score_mathvista(text, "145°", "multi_choice", "text", CH)["correct"], text
    assert not score_mathvista("</think>The answer is B", "145°", "multi_choice", "text", CH)["correct"]
    # Letters inside the reasoning do not count; only the final answer region does.
    assert score_mathvista("<think>maybe C</think>B", "145°", "multi_choice", "text", CH)["pred"] == "140°"
    assert score_mathvista("</think>It is 145°", "145°", "multi_choice", "text", CH)["correct"]   # choice text


def test_free_form_numbers_and_lists():
    assert score_mathvista("</think>so the total is 1,000", "1000", "free_form", "integer")["correct"]
    assert not score_mathvista("</think>about 999.6", "1000", "free_form", "integer")["correct"]
    assert score_mathvista(r"</think>\boxed{1.23}", "1.2", "free_form", "float", precision=1.0)["correct"]
    assert not score_mathvista("</think>1.26", "1.2", "free_form", "float", precision=1.0)["correct"]
    assert score_mathvista("</think>[0, 2, 1]", "[0, 2, 1]", "free_form", "list")["correct"]
    assert not score_mathvista("</think>no answer", "3", "free_form", "integer")["correct"]


def test_score_sample_dispatch():
    s = {"id": "mathvista-1", "answer": "3", "meta": {"question_type": "free_form", "answer_type": "integer"}}
    assert score_sample(s, "</think>3")["correct"]
    ss = {"id": "sspro-4", "bbox": (100, 50, 40, 20), "meta": {"img_size": (1000, 500)}}
    r = score_sample(ss, "pyautogui.click(x=120, y=120)")
    assert r["correct"] and r["parsed"] and not r["hit_pixels"]

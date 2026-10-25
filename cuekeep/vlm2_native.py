"""Unpruned Qwen2.5-VL on VLM2-Bench (GC/OC).

Mirrors the official code/{gc,oc}/test/test_script_example/test_qwen2p5_7B_img_qa_*.py
(same messages, same "\n"+question prompt, model's default generation config) and adds
--model, --max_pixels, per-sample visual-token counts and latency. Output lines are a
superset of the official ones, so the official eval scripts run unchanged.
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="HF id or local path")
p.add_argument("--question_file", required=True)
p.add_argument("--image_folder", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--max_pixels", type=int, default=0, help="per-image cap, 0 = native; tokens/img ~= max_pixels/784")
p.add_argument("--max_new_tokens", type=int, default=512)
p.add_argument("--attn", default="flash_attention_2", help="sdpa if flash-attn is not installed")
p.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
args = p.parse_args()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(args.model)
merge = model.config.vision_config.spatial_merge_size ** 2

with open(args.question_file) as f:
    questions = [json.loads(line) for line in f]
if args.limit:
    questions = questions[: args.limit]

os.makedirs(args.output_dir, exist_ok=True)
name = os.path.basename(args.model.rstrip("/"))
out_path = os.path.join(args.output_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}_answers.jsonl")
cap = {"max_pixels": args.max_pixels} if args.max_pixels else {}

records = []
with open(out_path, "w") as fout:
    for q in tqdm(questions):
        paths = [os.path.join(args.image_folder, im) for im in q["images"]]
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": f"file://{path}", **cap} for path in paths],
            {"type": "text", "text": f"\n{q['question']}"},  # official: f"{system_prompt}\n{question}", system_prompt=""
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0].strip()
        per_img = (inputs["image_grid_thw"].prod(-1) // merge).tolist()
        rec = {**q, "model_answer": answer, "n_vis_tokens": sum(per_img),
               "vis_tokens_per_image": per_img, "latency_s": round(dt, 3)}
        if "edited_prompt" in q:  # official GC output renames this field
            rec["edit_prompt"] = q["edited_prompt"]
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        records.append(rec)

n = len(records)
is_tf = n and records[0]["gt_answer"] in ("T", "F")
print("SUMMARY " + json.dumps({
    "file": out_path,
    "n": n,
    "vis_tokens_mean": round(sum(r["n_vis_tokens"] for r in records) / n, 1),
    "vis_tokens_max": max(r["n_vis_tokens"] for r in records),
    "latency_mean_s": round(sum(r["latency_s"] for r in records) / n, 3),
    "non_TF_answers": sum(r["model_answer"] not in ("T", "F") for r in records) if is_tf else None,
}))

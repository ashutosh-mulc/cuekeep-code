#!/bin/bash
# Gate 0 / step 1: unpruned Qwen2.5-VL on VLM2-Bench GC + OC, then the official evals.
# usage: bash run_gate0.sh <model> <max_pixels, 0 = native> <out_dir> [limit]
# env:   BENCH=<VLM2-Bench clone> (default ~/cuekeep/VLM2-Bench), ATTN=sdpa if flash-attn is missing
set -euo pipefail
MODEL=$1; MAXPIX=$2; OUT=$(realpath -m "$3"); LIMIT=${4:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${BENCH:-$HOME/cuekeep/VLM2-Bench}"  # cnt eval reads jsonl/oc/vanilla/oc_cnt.jsonl relative to repo root

gen() {  # question_file image_folder task max_new_tokens (official: 512 for GC, 2048 for OC)
  python "$HERE/vlm2_native.py" --model "$MODEL" --max_pixels "$MAXPIX" --attn "${ATTN:-flash_attention_2}" \
    --limit "$LIMIT" --question_file "$1" --image_folder "$2" --output_dir "$OUT/$3" --max_new_tokens "$4"
}
gen jsonl/gc/vanilla/gc_mat.jsonl data/gc/processed    mat 512
gen jsonl/gc/vanilla/gc_trk.jsonl data/gc/processed    trk 512
gen jsonl/oc/vanilla/oc_cpr.jsonl data/oc/original_obj cpr 2048
gen jsonl/oc/vanilla/oc_cnt.jsonl data/oc/original_obj cnt 2048
gen jsonl/oc/vanilla/oc_grp.jsonl data/oc/original_obj grp 2048

python code/gc/eval/eval_tf_batch_pair_3acc.py               --input_folder "$OUT/mat"
python code/gc/eval/eval_tf_batch_pair_3acc.py               --input_folder "$OUT/trk"
python code/oc/eval/eval_cpr/eval_comp_tf_pair_batch_3acc.py --input_folder "$OUT/cpr"
python code/oc/eval/eval_cnt/acc_with_cnt_score.py           --input_folder "$OUT/cnt"
python code/oc/eval/eval_grp/eval_acc_batch_classify_mcq.py  --input_folder "$OUT/grp"

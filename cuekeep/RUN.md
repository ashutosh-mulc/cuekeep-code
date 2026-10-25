# Gate 0, step 1: unpruned Qwen2.5-VL on VLM²-Bench GC + OC

- `vlm2_native.py`: one model on one task. It mirrors the official VLM² Qwen script and also logs token counts and latency.
- `run_gate0.sh`: runs all 5 tasks, then the official evals. It works from any folder; `BENCH` points to the VLM2-Bench clone.

These scripts are untested, so the smoke test is the first real test.

## 1. Setup (on a node with internet; ~31 GB of data and weights)

```bash
conda create -n cuekeep python=3.10 -y && conda activate cuekeep
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128   # cu126 if CUDA is older
pip install transformers==4.57.0 accelerate qwen-vl-utils==0.0.14 numpy Pillow tqdm
pip install flash-attn --no-build-isolation      # optional; if it fails, set ATTN=sdpa in the jobs

export HF_HOME=/big/disk/hf
mkdir -p ~/cuekeep && cd ~/cuekeep
unzip cuekeep_gate0.zip                          # gives ~/cuekeep/cuekeep/{vlm2_native.py,run_gate0.sh,RUN.md}
git clone https://github.com/vlm2-bench/VLM2-Bench.git && cd VLM2-Bench
huggingface-cli download Sterzhang/vlm2-bench vlm2-bench_dataset.zip --repo-type dataset --local-dir .
unzip -q vlm2-bench_dataset.zip                  # jsonl/ and data/ must end up at the repo root
ls jsonl/gc/vanilla jsonl/oc/vanilla; ls data/gc/processed | head -3; ls data/oc/original_obj | head -3
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

## 2. Jobs (one line per queue job)

Set the same `HF_HOME` in each job. Add `HF_HUB_OFFLINE=1` if compute nodes have no internet, and
`BENCH=/path/to/VLM2-Bench` if the clone isn't at `~/cuekeep/VLM2-Bench`.

```bash
conda activate cuekeep && cd ~/cuekeep/cuekeep
bash run_gate0.sh Qwen/Qwen2.5-VL-3B-Instruct 0      ~/cuekeep/results/smoke 10          # smoke test, minutes
bash run_gate0.sh Qwen/Qwen2.5-VL-7B-Instruct 0      ~/cuekeep/results/gate0/7b_native   # R1: reproduce paper
bash run_gate0.sh Qwen/Qwen2.5-VL-3B-Instruct 0      ~/cuekeep/results/gate0/3b_native   # R2: floor check
bash run_gate0.sh Qwen/Qwen2.5-VL-3B-Instruct 200704 ~/cuekeep/results/gate0/3b_256tok   # R3: 256 tok/img
bash run_gate0.sh Qwen/Qwen2.5-VL-3B-Instruct 802816 ~/cuekeep/results/gate0/3b_1024tok  # R4: optional
```

- **Use a fresh output folder every run.** The evals read every `*.jsonl` in a folder, so re-using one mixes runs.
- **Estimating run time:** each run is 2,240 questions. Take the smoke test's `latency_mean_s` and multiply by 2,240.
- **R1 should land within about ±2 points** of VLM² Table 1: Mat 35.91 · Trk 43.38 · Cpr 71.39 · Cnt 41.72 · Grp 47.50.

## 3. Send back

- The output of `grep SUMMARY <job log>` for each run (5 lines per run).
- The accuracy lines printed by the 5 eval scripts.
- If anything fails, the full traceback.

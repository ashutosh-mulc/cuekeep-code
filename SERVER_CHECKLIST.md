# Server checklist (GPU agent)

Rewritten 2026-11-12 around staged gates, for the agent working on the NVIDIA GPU server.

The work is ordered so that the cheapest test that could kill the idea runs first. Small **gates**
on fixed dev-split samples decide go / no-go, and every large run or large piece of engineering
waits for its gates. Read [research/experiment_design.md](research/experiment_design.md) before
starting: it holds each gate's sample, pass thresholds, on-fail action and what it unlocks.

## How to use this file

- **Read first:** `CLAUDE.md`, `research/experiment_design.md`, `research/method.md`,
  `research/decisions/0004-…` and `0005-…`.
- **Tick items in this file** as `- [x] … — YYYY-MM-DD · run: runs/E###/<ts> · commit: <sha>`. If an
  item fails, leave it unticked and add a `FAILED:` line under it saying what happened and where the log is.
- **Deciding a gate.** Each gate runner writes its verdict into `summary.json` as `gate/pass` and
  `gate/c*`. Judge a gate only by those pre-registered criteria. Then record it:
  `python scripts/gate.py set G# passed|failed --evidence <run dir> --note "<the numbers>"`,
  fill in the experiment README's *Gate decision* section, and commit.
- **A failed gate is a result.** Record it as failed, follow its on-fail action in
  `experiment_design.md`, and do not rerun with different thresholds or samples to make it pass.
- **The runner enforces gates.** `scripts/run.py` refuses a config whose `requires_gates` have not
  passed. Do not use `--override-gates` unless an item here says to. You may never set `waived`.
- **(human)** means: stop and ask the human before going further. Every failed gate whose on-fail
  action says (human) is one of these. So is any wish to change a threshold or a method default
  (`configs/method/cuekeep_v0.yaml`).
- **Log every session** in `research/notebook/YYYY-MM-DD.md`. Every number written anywhere must come
  from a `runs/` directory. Commit after each finished item, with a clean tree before every launch.
- **Never edit a Pre-registration section** once a run of that experiment exists.
- **Cost.** Before each launch, write its estimated GPU-hours in the notebook (samples × seeds ×
  median trace ÷ tokens/s, from G1). Any single launch above 24 GPU-hours is (human).

Gate order: `G0 → G1 → gate dump → G3 and G2 → G4 → G7`, with G5 and G6 running alongside once G1
has passed.

---

## Phase 0: Environment (gate G0)

- [ ] **Code is current.** `git log` includes the staged-gate commit, the tree is clean, and these exist:
      `research/experiment_design.md`, `experiments/gates.yaml`, `scripts/gate.py`.
      If they are missing, stop: the server has older code.
- [ ] **GPU and driver.** Run `nvidia-smi` and record the GPU model, count, memory, driver version, and
      the highest CUDA version the driver supports. Choose `CUDA_TAG` (e.g. `cu128`) to fit.
- [ ] **Storage.** Set `HF_HOME` on a disk with at least 60 GB free (weights about 9 GB, datasets,
      a gate dump of at most 3.5 GB, a main dump of at most 27 GB). Record the path.
- [ ] **Environment.** `CUDA_TAG=<tag> bash scripts/setup_server.sh` (needs Python ≥ 3.10), then
      `source .venv/bin/activate`. Write the VLMEvalKit commit it prints into `third_party/README.md`.
- [ ] **CPU tests on the server.** `pytest -q` must show 45 passed.
- [ ] **Env check.** `python scripts/env_check.py --model`; paste the JSON into the notebook. Required:
      `cuda_available`, `bf16_matmul_ok`, `sdpa_float_mask_ok` true; the model shows layers 36,
      heads 32, kv_heads 8, head_dim 128, spatial_merge 2, deepstack_layers [5, 11, 17].
      Record `h2d_pinned_GBps` under Q4 in `research/open_questions.md`.
- [ ] **CUDA cache tests.** Write `tests/gpu/test_cache_cuda.py` (`@pytest.mark.gpu`, skipped without
      CUDA): port `test_cache_reproduces_full_attention_when_all_committed`,
      `test_fetch_is_invisible_until_committed_at_lookahead` and
      `test_eviction_restores_standin_immediately_and_drops_pending` from `tests/test_paged_cache.py`
      to `device="cuda"`, and assert the host tensors are pinned. `pytest -q -m gpu` must pass.
- [ ] **Timing hook.** Through `cuekeep.models.get_adapter`, generate on one MME sample with
      `max_new_tokens=32`. `prefill_seconds` and `decode_peak_bytes` must not be None. If transformers
      rejects the duck-typed streamer, subclass `BaseStreamer` in `PhaseTracker` and rerun.
- [ ] **Decide G0.** All items above ticked:
      `python scripts/gate.py set G0 passed --evidence research/notebook/<date>.md --note "<env summary>"`.

## Phase 1: Premise (gate G1, experiment E000)

- [ ] **First run.** `GPUS=0 bash scripts/launch.sh experiments/E000-env-smoke/config.gate.yaml`.
      The ScreenSpot-Pro entry is *expected* to log a `load_error` (its format is not pinned yet).
- [ ] **Pin ScreenSpot-Pro.** From the columns in the `load_error`:
      find a source that loads with annotations (`Voxel51/ScreenSpot-Pro` is FiftyOne-format); set
      `hf_id`, `question_field`, `bbox_field`, `bbox_format` and `bbox_units` in
      `config.gate.yaml`; confirm the box format by drawing the box on 3 screenshots (save them in
      `experiments/E000-env-smoke/results/`); rerun. If no annotated source loads through `datasets`: (human).
- [ ] **Truncation.** If `gate/needs_full_length_rerun` is true, rerun with
      `model.max_new_tokens=40960 --tag gate-full-length` and judge that run.
- [ ] **Decide G1** from `gate/*` in `summary.json`, and record E000 (README runs, results, gate
      decision; provenance; index status; notebook). Also write down, for later cost estimates:
      decode tokens/s and the median trace length per dataset.
      - `gate/c1_token_order` false: stop, (human).
      - `gate/c2_long_traces` or `gate/c3_visual_share` false: (human).

## Phase 2: One dump, two gates (G3 and G2)

- [ ] **MathVista loader.** Add `src/cuekeep/data/mathvista.py`, following `mme.py` (`take_usable`,
      `role` / `dev_size` support), register it in `load_samples`, and add a CPU test.
- [ ] **Dump script.** Write `scripts/dump_visual_kv.py`, producing the format documented in
      `src/cuekeep/analysis/dumps.py` (one `.pt` file per sample × layer):
      - Generate each sample with full KV and the model's own sampling (per-sample seed as in E000).
      - Per (sample, layer): `k`, `v` of the visual span from the cache (post-RoPE), stored once.
      - Per sampled step: `q` (post-norm, post-RoPE, all 32 heads), and the text summary `other_lse`,
        `other_out`: the log-sum-exp of the query's logits over every non-visual key visible at that
        step, and those keys' attention-weighted mean value. Do **not** store the text KV.
      - Also store `attn_out`, the model's real per-head attention output, for the capture check.
      - Take arguments for the sample list, layers, number of spread steps, and dense windows.
- [ ] **Capture the gate dump.** The first 8 dev samples of ScreenSpot-Pro and of MathVista
      (`role: dev, dev_size: 150, limit: 8`); layers 2, 5, 11, 17, 24, 35; 64 steps spread over each
      trace; and for the first 2 samples of each dataset, a dense window of 256 consecutive steps
      starting 25% into the trace, saved as separate units. For those same 2 + 2 samples, also save
      pre-RoPE keys (`rope: "pre"`). Check the size is at most a few GB.
- [ ] **Run G3 first** (attention shift; it is the cheapest way the idea can die):
      `python scripts/run.py experiments/E003-attention-shift/config.gate.yaml dump_path=<dump dir>`.
- [ ] **Run G2** (stand-in fidelity):
      `python scripts/run.py experiments/E001-pooled-attention-fidelity/config.gate.yaml dump_path=<dump dir>`.
      If `gate/c1_capture` is false, the dump is wrong: fix the script, recapture, and rerun **both** gates.
- [ ] **Decide G3 and G2** from their `gate/*` values; record both experiments.
      - G3 `gate/kill` (gap < 0.05 everywhere): record failed, then stop everything except G5 and G6
        and (human). Do not write the paging hook.
      - G3 `gate/marginal`, or `gate/c2_lag1_tracks` false: (human).
      - G2 criterion 3 fails: rerun G2 on the pre-RoPE units once, report both, then (human).
      - G2 criterion 2 fails: (human).

## Phase 3: Alongside (gates G5 and G6; start once G1 has passed)

- [ ] **G5 overhead microbenchmark.**
      `python scripts/run.py experiments/E005-paging-overhead-microbench/config.gate.yaml bench.baseline_decode_ms_per_token=<ms from E000>`.
      Decide G5 from `gate/pass` (needs `gate/valid` true). On fail: optimise as its README lists,
      rerun; after two failed rounds, (human).
- [ ] **Benchmark names.** Check every name in `configs/eval/thinking_suite.yaml` against the installed
      VLMEvalKit; fix them. A missing benchmark: (human) before substituting.
- [ ] **VLMEvalKit wrapper.** A model class around `Qwen3VLAdapter` with per-sample seeds, the model's
      own decoding, `max_new_tokens=40960`, the `GenerationRecord` fields per sample, and a dev / test /
      all selection using `cuekeep.data.split_indices` on each benchmark's row order. Implement
      `experiments/E002-full-kv-reference/run.py`, writing `gate/*` for the criteria in its README.
- [ ] **G6 reference sanity.** `GPUS=<n> bash scripts/launch.sh experiments/E002-full-kv-reference/config.gate.yaml`
      (150 dev samples per benchmark, 1 seed). Decide G6. On fail: fix prompting or answer extraction,
      rerun. Write the GPU-hour estimate for E002's main tier in the notebook.

## Phase 4: Paging in simulation (gate G4; held until G2 and G3 have passed)

- [ ] **Masked-simulation hook.** Implement paging inside the model with all KV kept on the GPU:
      stand-ins as extra keys with the log m bias, residency through the additive attention mask
      (plan in `src/cuekeep/cache/hf_adapter.py`; reference semantics `pooling.attend_mixed`).
      It must support teacher-forced replay with a residency set per position.
- [ ] **Prune-once baseline** in the same hook: top tokens by attention at the end of the prompt,
      fixed, the rest dropped, matched in entry count to CueKeep's budget.
- [ ] **Implement and run E004.** Write `experiments/E004-teacher-forced-paging-fidelity/run.py` to its
      README, then run `config.gate.yaml`. The equivalence criterion (100% budget) must hold before
      any other number from the hook is read.
- [ ] **Decide G4**; record E004. Criterion 2 fails: (human). Criterion 3 alone fails: tune g and the
      budget on dev, report, then (human).

## Phase 5: Accuracy pilot and main tiers (held as shown)

- [ ] **G7 pilot** (needs G3, G4, G6). Implement `experiments/E006-accuracy-memory-tradeoff/run.py`,
      choose the 2 benchmarks from E002's gate run, run `config.pilot.yaml`, decide G7. On fail: (human).
- [ ] **Main dump** (needs G2 and G3). 30 + 30 *test* samples, 13 layers, 64 steps, dense windows for
      5 + 5 samples, post- and pre-RoPE. Check the size is at most about 27 GB.
- [ ] **E001 main and E003 main** on the main dump (`config.main.yaml` of each). Record results
      against their main-tier success criteria; update Q1, Q2, Q3, Q10, Q12.
- [ ] **E002 main** (needs G6): (human) approves the cost estimate first. Then 3 seeds, full suite.
- [ ] **E006 main** (needs G7): implement the remaining baselines first; (human) approves the cost.

## Phase 6: Real offload (held until G4 and G5 have passed)

- [ ] **`install_paged_attention` with real offload:** ingest into `PagedVisualCache`, physically delete
      the visual spans from the DynamicCache, compute `other_lse`, then `step → commit → attention_inputs`.
- [ ] **Equivalence test** `tests/gpu/test_hf_equivalence.py`: with every region committed and
      `lookahead: 0`, the first 64 greedy tokens on 5 samples match full KV, and logits match within
      bf16 tolerance. It must also match the masked simulation at a 25% budget.
- [ ] **First real memory numbers:** `PagedVisualCache.gpu_bytes()` against full visual-KV bytes, and
      the decode-phase peak for both, on one ScreenSpot-Pro sample. Record in the notebook.

## Always

- When a run finishes: update the experiment README (runs, results, gate decision), `provenance.yaml`,
  the index status in `experiments/README.md`, the gate ledger, and the notebook; then commit.
- New risks or surprises go in `research/open_questions.md`, with the experiment that would resolve them.

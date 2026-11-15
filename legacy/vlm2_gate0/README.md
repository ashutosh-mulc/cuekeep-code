# legacy/vlm2_gate0: Gate 0 scripts from the earlier pruning direction

Before the move to pooled-KV paging, CueKeep studied how far visual tokens can be pruned before a
VLM stops linking cues across images. That work used Qwen2.5-VL on VLM²-Bench, with TPRL as the RL
baseline. This folder holds that direction's Gate 0, step 1: unpruned Qwen2.5-VL baselines on
VLM²-Bench GC + OC (runs R1–R4), plus its working notes.

| File | What it is |
|---|---|
| `vlm2_native.py` | One model on one VLM²-Bench task. It mirrors the official VLM² Qwen test script and also logs visual-token counts and latency |
| `run_gate0.sh` | Runs all five tasks (Mat, Trk, Cpr, Cnt, Grp), then the official eval scripts |
| `RUN.md` | Setup, job lines for R1–R4, and what to send back |
| `TODO.md` | Notes for that direction as of 2026-10-26: research gap, TPRL code findings, hypotheses H1–H6, reading list, stats protocol, open decisions |

## Where they came from
- Branch `research/cuekeep` of `ashutosh-mulc/cuekeep-code`, commit `9585ad2` (2026-10-26).
- That commit was cherry-picked into `main` as `c73fd07`, keeping its original author and date.
  The files were then moved here unchanged.
- The `research/cuekeep` branch itself is still on GitHub.

## Status
- **Never run.** RUN.md says these scripts are untested.
- **Not part of the gated pipeline.** They don't go through `scripts/run.py` or `experiments/gates.yaml`,
  so under [CLAUDE.md](../../CLAUDE.md) their numbers can't be reported. To use them for the paper,
  register an experiment with `scripts/new_experiment.py` and launch it through `scripts/run.py`.
- **Different stack.** They need Qwen2.5-VL and transformers 4.57 in their own conda env (`cuekeep`,
  see RUN.md). The current repo uses Qwen3-VL-Thinking and transformers 5.x.

## Notes on the old paths
- RUN.md uses the old layout. Read `~/cuekeep/cuekeep/` as `legacy/vlm2_gate0/` in this repo, and
  skip the `unzip cuekeep_gate0.zip` step.
- `run_gate0.sh` finds `vlm2_native.py` next to itself, so it runs from here.
- `BENCH` points to a VLM2-Bench clone, for example `third_party/src/VLM2-Bench` (gitignored). The
  pinned commit and license are in [third_party/README.md](../../third_party/README.md).
- TODO.md's visibility warning is about the old local checkout's public `origin`. This repo is private.

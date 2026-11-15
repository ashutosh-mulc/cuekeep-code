# CLAUDE.md: working conventions for CueKeep

Research repo: **pooled-KV paging for visual tokens in thinking VLMs** (Qwen3-VL-Thinking), targeting
ICML 2027. Start with `research/overview.md` and `research/method.md`.

## On the GPU server

The server **cannot push to git**: it runs only what is prepared and pushed from the laptop. Follow
[SERVER_CHECKLIST.md](SERVER_CHECKLIST.md): pull a clean commit, run the prepared commands, never edit
tracked files there, and report by encoding the `runs/` folders (`encode_dir.sh`). Gate decisions,
experiment records and the notebook are written on the laptop from the decoded runs.

## On the laptop (preparing work for the server)

- Everything that goes to the server must be tested first. Use a local environment that matches the
  server (Python 3.12, torch 2.15.0, transformers 5.23.0; CPU), and the tiny Qwen3-VL
  (`python -m cuekeep.testing.tiny_qwen3vl <dir>`) to run model-facing code end to end.
  `pytest -q` covers the math, loaders, the tiny-model integration tests (`-m integration`) and
  accelerator tests (`-m gpu`; MPS on a Mac, CUDA on the server).
- Test tolerances must follow from the computation's precision, not be loosened until a test passes.
  Check fragile assertions across many seeds, not just one.
- The GPU is shared: timing gates measure their own baseline in-process and compare medians.

## Where things go

- **Code runs on a remote NVIDIA GPU server, not this laptop.** Library code must import without
  CUDA. GPU-only paths are guarded with `torch.cuda.is_available()`, and the `transformers` imports
  are lazy. Only the math in `src/cuekeep/cache/pooling.py`, the paging bookkeeping and the
  tracking code are tested locally.
- **Experiments are staged.** Small gates on fixed dev-split samples hold back the main tiers
  (`research/experiment_design.md`, ledger `experiments/gates.yaml`). `scripts/run.py` refuses a config
  whose `requires_gates` have not passed. Decide a gate only from its pre-registered criteria, record
  it with `scripts/gate.py`, and treat a failed gate as a result. Only the human waives a gate or
  changes a threshold. Tune only on the dev split. The general pattern, with templates, is the
  user-level `staged-experiments` skill; use it when designing any new experiment here.
- New experiment: `python scripts/new_experiment.py "title"`, then fill in the **pre-registration
  before running**. Never edit a pre-registration after results exist. Add to Results/Conclusion
  instead.
- Every run goes through `scripts/run.py` (or `scripts/launch.sh` on the server), which writes
  `runs/<E###>/<ts>/`. Do not report numbers that lack a run directory.
- When a run lands: update the experiment README's Runs table, `provenance.yaml`
  (`executed`, `result_file`, `negative_results`), the index status in `experiments/README.md`,
  and today's `research/notebook/YYYY-MM-DD.md`.
- Choices that would be costly to reverse go in `research/decisions/NNNN-*.md`. New risks go in
  `research/open_questions.md`.
- Third-party code: record the license and the pinned commit in `third_party/README.md` *before*
  copying anything.

## Paper pipeline (ARS `academic-pipeline`)

- Stage outputs go in `paper/stage*/`, using the ARS version labels (`paper_draft_v2.md`, ...).
  Never overwrite a version.
- Passport: always pass the explicit path `paper/passports/cuekeep_passport.yaml`.
- At the experiment-intake question, the user answers in their own words. Never pre-fill
  `experiment_intake_declaration`. Supply provenance from `python scripts/export_provenance.py`.
- Literature corpus: one note per paper in `research/literature/papers/<citekey>.md` (Convention A
  frontmatter, metadata copied from the paper itself), built with `bash scripts/build_corpus.sh`.
  Never invent authors, venues or years.

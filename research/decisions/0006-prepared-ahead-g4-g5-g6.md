# 0006: Build G4–G6 code ahead of their gates; how G4, G5 and G6 are run

- **Date:** 2026-11-16
- **Status:** accepted (one item awaits the human: G5's judging baseline)

## Context
The server cannot push and runs only what is prepared on the laptop. It gets one working session per
day. The staged design ([experiment_design.md](../experiment_design.md)) holds engineering until gates
pass, to avoid wasted work. With a day's latency per round trip, waiting would cost far more than the
code. The human asked for everything that can be prepared to be ready for the next session.

## Decisions

1. **Code is built ahead of its gates; runs stay gated.** The masked-simulation hook (E004), G5 round 1
   and the E002 harness were written and tested on 2026-11-16, before G2 and G3 have run.
   `scripts/run.py` still refuses E004 until G2 and G3 pass. If G3 fails, the E004 code is shelved
   unused. That is an accepted cost.

2. **G5 round 1 (E005).**
   - The cache keeps persistent device buffers, so attention inputs are views (round 0 rebuilt them
     every step, about 4 ms).
   - Scoring is batched across layers, with one host sync per refresh (`queue_query` + `end_step`).
     Round 0 synced once per layer, and on a shared GPU each sync waits behind other jobs.
   - `SlotPool.plan` works on plain Python lists: about 23 µs per call, down from about 140 µs.
   - On the laptop GPU (MPS; not representative of CUDA copies), a step at refresh_every 4 went from
     15.9 ms to 6.8 ms.
   - Round 1 also measures a **baseline in the same process**. The real model decodes at the gate's
     visual-token count, interleaved with paging steps, and medians are compared. Both verdicts are
     reported. `gate.judge` stays `preregistered` (E000's mean, 42.9 ms) **until the human decides**
     whether `in_process` should judge, since this changes the judging after a failure.

3. **G6 (E002) uses an in-repo harness on MathVista-mini and ScreenSpot-Pro**, not VLMEvalKit's full
   suite. The pre-registration was revised before any run. Reasons:
   - VLMEvalKit's MathVista scoring needs a GPT judge.
   - Its ScreenSpot-Pro path does not convert Qwen3-VL's 0–1000 coordinates.
   - G7 needs exactly one reasoning-heavy and one high-resolution benchmark.
   Scoring is by rule (`cuekeep.eval.scoring`). The references are the official Qwen3-VL-4B-Thinking
   scores from the model card's benchmark table: MathVista-mini 79.5, ScreenSpot-Pro 49.2. Other suite
   benchmarks are added later, with their own loaders.

4. **G4 (E004) details fixed before any run.**
   - **Reference.** The reference for every comparison is full KV through the simulation's own
     float32 path (`all_resident`), so condition differences come from paging, not from kernel
     numerics. Criterion 1 (the hook check) compares that path with the untouched model.
   - **Gate point.** Criteria 2–3 are read at the method's region size, g = 4, and budget 0.25.
   - **CueKeep in simulation** follows the batched protocol of G5 round 1. Decisions are made after
     step t's attention, with step t's query. Fetches are visible from t + lookahead and evictions
     from t + 1. At the end of prefill, all slots are filled from the last prompt query's scores
     (`prefill_fill`, a new method knob, default on).
   - **Prune-once** keeps as many exact tokens as CueKeep holds entries (exact tokens + stand-ins), as
     pre-registered.

## Consequences
- Tomorrow's session can run G5 round 1 and G6 alongside the gate dump, and G4 the same day if G2 and G3 pass.
- E006 (the accuracy pilot) still needs free-running paged generation, which is not yet written.

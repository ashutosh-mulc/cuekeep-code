# Roadmap

_Last updated: 2026-11-12 · Target: ICML 2027 submission (late Jan 2027; confirm the exact date when the call is out)_

The order of work follows the gates in [experiment_design.md](experiment_design.md). A milestone is
done when its gates are passed in [experiments/gates.yaml](../experiments/gates.yaml), not when its
code exists. Week numbers are a plan, not a commitment; a failed gate changes them.

Status legend: ☐ not started · ◐ in progress · ☑ done · ✗ dropped

## M0: Scaffolding (week of 2026-11-11)
- ☑ Fresh repo with the research-log and experiment structure
- ☑ Code review fixes ([decisions/0004](decisions/0004-review-fixes-scoring-memory-fetch.md)); staged-gate design ([decisions/0005](decisions/0005-staged-gates.md))
- ☐ Check Reroute's license; pin commits for Reroute, ShadowKV and kvpress in [third_party/](../third_party/README.md)

## M1: Is the premise true? (week 1) — gates G0, G1, G2, G3
- ☐ **G0** server environment; GPU-only code paths tested
- ☐ **G1** premise census (E000): trace lengths, visual share, token order
- ☐ Gate dump (one capture, 8 + 8 dev samples)
- ☐ **G3** attention shift (E003). If this fails, stop and redirect before any paging code
- ☐ **G2** stand-in fidelity (E001)
- ☐ In parallel: **G5** overhead microbenchmark (E005); VLMEvalKit wrapper and **G6** reference sanity (E002)

## M2: Does paging work in simulation? (weeks 2–3) — gates G4, G7
- ☐ Masked-simulation attention hook (held by G2 + G3)
- ☐ **G4** paging fidelity under teacher forcing (E004), with a prune-once baseline
- ☐ **G7** free-running accuracy pilot on dev (E006 pilot)
- ☐ Main dump; E001 main and E003 main (fidelity numbers and the attention-shift figure)

## M3: Main results (weeks 4–8) — held by G6, G7
- ☐ E002 main: full-KV reference, 3 seeds (cost approved by the human)
- ☐ Remaining baselines in the same hook: FastV, PDrop, Quest-over-regions, ShadowKV-style, VisionPulse-style, DSTP-style, SnapKV
- ☐ E006 main: accuracy vs visual-KV bytes, full suite, 3 seeds

## M4: Systems numbers and ablations (weeks 6–10) — held by G4, G5
- ☐ Real offload: pinned host store, async fetch, physical deletion from the HF cache; equivalence test against full KV
- ☐ Decode ms/token, total response time, PCIe bytes per token, visual-KV GPU bytes, batch capacity
- ☐ Ablations: log m bias, covariance rank and mode, g, refresh interval, fetch budget, hysteresis
- ☐ Optional: Vortex or SGLang serving port (after a 1-day smoke test)

## M5: Paper (weeks 10–11)
- ☐ Draft through the ARS pipeline (`paper/`), figures, appendix with the bound; internal review

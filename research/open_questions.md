# Open questions and risks

Each item names the experiment or action that resolves it. Close an item by striking it through and
linking the evidence.

| # | Question / risk | Severity | Resolved by | Status |
|---|---|---|---|---|
| Q1 | Does the log m pooled stand-in approximate attention well on *real* Qwen3-VL keys, given that RoPE spreads keys within a region? Pool pre- or post-RoPE? | High | E001 (gate G2, then main) | open |
| Q2 | Does the covariance correction track true region mass better than the plain pooled logit? What rank r is enough? | High | E001 main | open |
| Q3 | Do the regions a thinking trace needs actually shift over decoding (H2)? | High | E003 (gate G3, then main). A failed G3 redirects the project | open |
| Q4 | Can PCIe fetch latency be hidden with a 1-step lookahead at batch size 1? (Protocol implemented 2026-11-12; untested on GPU) | Medium | E005 (gate G5); `env_check.py` H2D bandwidth | open |
| Q5 | Does degraded perception lengthen traces enough to cancel the per-token savings (the ViMoD effect)? | Medium | E006 pilot (G7) trace-growth criterion; total response time in every table | open |
| Q6 | Reroute license: can we copy its code, or only take inspiration? | Medium | Check the repo | open |
| Q7 | Does Vortex/SGLang support a Qwen3-VL decode path? | Low (optional) | 1-day smoke test | open |
| Q8 | Will ViMoD, VisionPulse or SparseVILA release code before reviews? | Medium | Re-check Jan 2027 | open |
| Q9 | The exact demand-score definition in the original design notes | Medium | Compare with the design session | open |
| Q10 | Is attention mass the right fetch priority? On synthetic keys, every score variant ranks regions by the oracle benefit of fetching only weakly (Spearman 0.2–0.57) | High | E001 `benefit/*` on real keys; E004 CueKeep vs oracle paging | open |
| Q11 | How many seeds does sampling decoding need for stable method comparisons? | Medium | E002 main seed variance | open |
| Q12 | Does per-head share normalisation beat raw-max aggregation on real heads? (It wins on a constructed loud-head case; no difference on synthetic data without loud text heads) | Medium | E001 `agg/*` on real keys | open |
| Q13 | Are the gate thresholds in `experiment_design.md` set at the right level? They were chosen before any data existed | Medium | Review after G1–G3; changes need the human and a note in decisions/0005 | open |

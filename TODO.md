# CueKeep TODO

Working notes for CueKeep (plan and hypotheses: `../README.md`). Last updated 2026-10-09.
Compute: 1× H100 80 GB, queued jobs, 24 h max per job (more GPUs possible with advance notice).

**Research gap (agreed 2026-10-09).** The problem isn't pruning *across* images. It's how far visual tokens can be pruned
before the VLM's reasoning suffers, here on cue linking. Pruners are designed and tested on single-image tasks, where
keeping the tokens that match the question is enough. Linking cues across images depends on small details the question
usually doesn't name (a swapped attribute, a removed object, a distractor's logo). Nobody has measured whether pruners keep
those details at a given budget, or why they drop them.
Dropped: O3 (positions across images) as a direction, keeping one guardrail (original position IDs in every method);
H3 (budget allocation across images: off-focus, and AVTP covers it).

> **Visibility:** this repo is a GitHub fork of a public repo, and forks of public repos are always public.
> Don't push this file or any CueKeep code to `origin` before the deadline. Use a private remote.

---

## Now: Gate 0 step 1, unpruned baselines on VLM² (no TPRL code involved)

Scripts: `cuekeep/vlm2_native.py` (mirrors the official VLM² Qwen script, plus `--model`, `--max_pixels`,
token counts and latency) and `cuekeep/run_gate0.sh` (GC+OC generation, then the official evals).

- [ ] Env, data and models on the server
- [ ] Smoke test: 3B, 10 questions per task
- [ ] **R1** Qwen2.5-VL-7B, native resolution. Must reproduce VLM² Table 1:
      Mat 35.91 · Trk 43.38 · Cpr 71.39 · Cnt 41.72 · Grp 47.50. This validates data, prompts and eval.
- [ ] **R2** Qwen2.5-VL-3B, native resolution. Floor check.
- [ ] **R3** 3B at 256 tok/img (`max_pixels=200704`). The no-prune upper bound at TPRL's Qwen budget.
- [ ] **R4** 3B at 1024 tok/img (`max_pixels=802816`). Optional middle point.
- **Decision after R2/R3:** if 3B Mat/Trk paired accuracy is within ~5 pts of chance (25), GC can't
  show a pruning drop. Options: use 7B for the inference-only analysis (breaks the fixed-backbone
  guardrail), or centre H1 on OC.

Notes:
- The official GC eval checks `model_answer == gt_answer` exactly, with no normalisation, while the OC-Cpr eval
  uses a regex. So a 3B model answering "T." scores as wrong on GC. The `SUMMARY` line prints `non_TF_answers`.
  Report strict official scores, plus our own single-parser scores later.
- Their generation uses the model's default `generation_config` (not explicit greedy). We switch to explicit greedy for
  the pruning experiments and re-run the no-prune reference with it.

---

## TPRL code findings (not fixed yet)

### Blocking (confirmed statically)
- [ ] `model/qwen_mllm.py:61`: IndentationError, so the Qwen backbone can't be imported.
- [ ] `pruner/rl_pruner.py:4`: `from .rl_networks` but the file is in `model/`. This breaks `train_lfd.py` and `train_ppo.py`.
- [ ] `main.py:15`, `train_ppo.py:139`: import `evaluator/`, deleted in commit `6617257`. `main.py` dies.
      In PPO the import is inside try/except, so the bbox reward is silently 0.0 every time.

### Likely crashes (from reading; fix regardless)
- [ ] `train_ppo.py:324`: `torch.stack` over states with different token counts (each step shrinks K; Qwen
      dynamic resolution also varies N across images).
- [ ] `train_ppo.py:355-364`: `actor.step()` updates the shared-attention weights in place, then `value_loss.backward()`
      runs through the same graph. Likely an in-place-modification RuntimeError.
- [ ] `train_ppo.py:335`: log-probs are recomputed without the `0.5^t` step discount used at sampling (`:211`), so
      the PPO ratio is wrong for steps ≥1. Entropy is also computed without the discount.

### Silent fidelity bugs
- [ ] **3-D M-RoPE lost:** only `inputs_embeds` reaches `generate`, so Qwen2.5-VL falls back to 1-D positions.
      Even "no pruning" isn't native Qwen2.5-VL.
- [ ] **Multi-image boundaries lost:** `qwen_mllm.py:45-48` cuts from the first to the last image token, dropping the
      `<|vision_end|><|vision_start|>` separators between images, and splices all visual tokens in as one block.
      (The processor itself handles multiple images; `get_image_features` returns a per-image tuple.)
- [ ] `topk(sorted=False)` (`rl_pruner.py:139`, `base_pruner.py:63`): kept tokens come out in arbitrary order.
- [ ] `train_ppo.py:149`: the reward is `gt in output` (substring), so a 'T'/'F' ground truth matches almost any output.
- [ ] `qwen_mllm.py:12`: no dtype. Load bf16 explicitly.

### Env / infra
- [ ] `config.py:7-9` hardcodes the authors' HF_HOME, `hf-mirror.com` and `CUDA_VISIBLE_DEVICES=3,5,6`.
- [ ] `requirements.txt` lacks `accelerate`, which `device_map="auto"` needs.
- [ ] `train_lfd.py` keeps all demo feature tensors in RAM and then `torch.save`s them. 50K demos × ~600 tokens × 2048-d
      × bf16 ≈ 100+ GB. Store kept indices and recompute features, use fewer demos, or use AE latents.
- [ ] No resume logic. With a 24 h job cap, PPO must resume from checkpoint (model + optimizers + epoch + RNG).

## Code vs paper (our "TPRL" baseline is a re-implementation; say so in the paper)
- No autoencoder (`USE_AUTOENCODER=False`, TODO stubs). The policy runs on raw 2048-d tokens. The paper also contradicts
  itself on AE dims (1024→256→64 in the text vs 4096→256 in the ablation).
- LfD demos come from `RandomPruner` (`train_lfd.py:232`); the paper uses VisPruner. Imitating random labels teaches a
  constant keep probability, so LfD is effectively a no-op. The schedule is 80/60/40% here vs 100→75→50% in the paper.
- VisPruner relies on ViT [CLS] attention, and Qwen2.5-VL's ViT has no CLS token, so it's unclear how the paper ran it on Qwen.
  We need another LfD teacher (FastV scores or DivPrune).
- Default dataset is ScreenSpot-Pro (GUI bbox). Of the paper's benchmarks, only an MME loader exists.
- Paper headline numbers use a **fine-tuned LLM** (Table 5); frozen gives 98.5–99.5% relative. No fine-tuning code.
  We compare against frozen TPRL.
- Reproduction target: TPRL Qwen2.5-VL-3B, MME at 144 tokens, 1399 (96.5% relative; upper bound at 256 tokens).
  This doubles as the single-image no-regression check.
- Paper inference uses a threshold, so the kept count adapts per sample. For equal budgets we use top-k for every method.

## Fix plan (when coding starts)
- Keep `model/rl_networks.py` and the PPO/GAE core (after the fixes above).
- New Qwen wrapper: keep all text and separator tokens. Compute 3-D `position_ids` with the model's `get_rope_index` on the
  full sequence, then gather the kept visual positions. Decode with a short manual greedy loop (answers are ≤ ~10 tokens).
- VLM² loader and one evaluator with a single regex parser (also report the strict official score).
- **Gate 0 acceptance:** with keep = 100%, the wrapper's answers are identical to native `generate` on ≥50 VLM² items.

---

## Hypothesis implications
- **H6 is probably false for TPRL by construction.** The query is the mean of the *raw, non-contextual* input embeddings over
  the whole prompt, system prompt included (`qwen_mllm.py:50-51`). "same" and "not the same" differ by one token
  in about 60, so the kept sets should be near-identical. Measure the Jaccard overlap anyway (cheap). It may differ for FastV.
- **New interpretability test (H5-adjacent):** does the policy use the query at all? Swap or zero the query and measure
  how much the kept set changes. If it barely changes, "language-guided" pruning is mostly visual saliency.
- **H2 is a clean single-variable change:** the policy's self-attention already sees all images' tokens jointly, so only
  the reward changes.
- **H4 masks:** VLM² ships no edit masks. Derive them from `|I_ori − I_edit|`, thresholded with morphological cleanup
  (diffusion edits shift pixels globally). Exclude environment edits. H4 is GC-only, since OC has no masks.
- **H1 floor risk:** 7B scores only 35.9 on Mat and 43.4 on Trk (chance 25). 3B isn't reported. Hence R2.
- **Absence (O2):** pruning doesn't hurt single-image absence (POPE "Is there a dog?" stays near vanilla). In VLM²
  add/remove, the model must check *what is now at that spot* in image 2 against image 1, and query-guided pruning has
  nothing to anchor on there. Test it by breaking H1 down by change type: add/remove should drop more than attribute.
- **VLM² Table 3 prior:** halving the resolution on 7B drops GC-Mat 35.9→25.1 and OC-Cpr 71.4→64.2, but OC-Cnt *rises*
  41.7→45.8. Detail matters for matching and comparison but not for counting.
- **Per-image token budget:** Qwen's default `max_pixels` allows thousands of tokens per image. Fix a per-image cap
  (TPRL used ~256 tokens per image).

## Reading list (pruned 2026-10-09; no more searching unless a specific decision needs it)

**Core (PDFs in the repo):**
- **TPRL**: the RL base.
- **IVC-Prune** (ICLR 2026, arXiv 2602.03060): the senior's reference. Training-free and inside the LLM
  (Qwen2.5-VL-3B: layer 22 of 32). Keeps position-only "implicit visual coordinate" tokens (10%) plus foreground tokens
  picked by value-vector similarity (40%), and keeps original position IDs. Shows that renumbering positions or merging
  tokens destroys grounding. Video is pruned per frame; no multi-image eval. Public code supports our backbone.
- **VLM²-Bench**: the evaluation.

**Skim one section each:**
- **AVTP** (EMNLP 2026 Findings, arXiv 2608.26806): the nearest prior evaluation of pruning on multi-image benchmarks
  (MuirBench, BLINK, NLVR2, ...), with per-image keep ratios. No VLM², no Qwen2.5-VL, no RL, no analysis of what's lost.
  Skim the results table.
- **Wen et al.** (ACL 2025 Findings, arXiv 2502.11501): does language guidance help pruning? The general form of O2.
  Skim the language-guidance section.

**Cite only (no reading):**
- **VTC-Bench** (ACL 2026, arXiv 2510.07143): downsampling beats many compression methods → we include a downsampling
  baseline (it comes free from Gate 0's R3/R4 resolution caps).
- **ToFu** (arXiv 2503.04444): training-free merging of look-alike tokens across the multi-image stream.
- **AOT** (CVPR 2026, arXiv 2603.01400): cross-frame merging for video. Its authors asked TPRL to cite it.

**Dropped as off-track (don't re-search):** AVAM, UniPruneBench, S²Prune, OccamToken, VDiff-Bench, QCPruner,
DUET-VLM, "When Token Pruning is Worse than Random", "Not All Redundant Tokens Are Alike", "Feather the Throttle".

## Stats protocol (fix before running H1)
- GC-Mat has 260 pairs, so SE ≈ 2.8 pts at p≈0.3 (95% CI ±5.6). Gate 2's "~2 pts" can't be resolved per subtask.
- Primary test: paired, per-item correctness across methods (McNemar), plus bootstrap CIs. Pool all GC+OC T/F pairs
  (840 pairs, SE ≈ 1.6) as the headline.
- Report kept tokens and latency in every table (guardrail).

## Open decisions
- [x] Private remote for this repo: `private` → ashutosh-mulc/cuekeep-code, branch `research/cuekeep` (2026-10-26).
      Pushing to `origin` is disabled locally (`git remote set-url --push origin no-push-public-fork`).
- [ ] Floor-effect fallback (after R2/R3)
- [ ] Attention heuristic and LfD teacher on Qwen: FastV (matches TPRL's Qwen table), DivPrune, or IVC-Prune (public code)
- [ ] Baseline set (decide after Gate 0): no-prune, downsampling (free from R3/R4), random, IVC-Prune and/or FastV, TPRL.
      That's over the 4-baseline cap, so pick.
- [ ] H2 training edits: exclude OmniEdit/ImagenHub. If using MagicBrush, perceptual-hash dedupe against VLM² GC
      (ImagenHub's editing set may draw on MagicBrush; unverified)

"""Masked simulation of region paging inside a Qwen3-VL forward (teacher-forced). Used by E004 (gate G4).

All KV stays on the GPU. For the paged layers, the attention of every decode query (positions
P … L-1, where P is the prompt length) is recomputed in float32 over

    [ non-visual keys ≤ p | exact visual tokens kept at that step | stand-ins of the other regions ]

with each stand-in entering as a single key (mean K, mean V) with a log m logit bias, as in
`pooling.attend_mixed`. Prompt positions keep the model's own (full) attention, as prefill does.
What each step keeps is decided by a policy (below). Comparing the next-token distributions of a
policy with those of full KV, under the same teacher-forced trace, isolates the effect of paging.

Policies (all return, per decode step t, which visual tokens are exact and which stand-ins are used):
  AllResident    everything exact: full KV through the same code path (the reference, and the hook check)
  StandinsOnly   no region exact, every region as its stand-in
  RandomRegions  a fixed random set of C regions exact, stand-ins for the rest
  Oracle         at each step, the C regions with the most true attention (heads summed) exact
  CueKeep        the slot-pool policy, run causally as in PagedVisualCache's batched protocol:
                 at the end of prefill the pool is filled from the last prompt query's scores;
                 after decode step t, the step-t query scores regions, and fetches become visible at
                 t + lookahead; evictions take effect from t + 1
  PruneOnce      FastV-style: the K visual tokens with the most attention from the last prompt query
                 are kept exact, the rest dropped (no stand-ins); K matches CueKeep's entry count
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..analysis.capture import _text_attention_modules
from ..cache.paged_visual_cache import SlotPool
from ..cache.pooling import demand_scores, exact_region_log_mass, pool_regions, region_log_shares


class LayerContext:
    """What a policy may look at, for one paged layer in one teacher-forced forward."""

    def __init__(self, layer: int, q: Tensor, k: Tensor, v: Tensor, visual: Tuple[int, int], prompt_len: int,
                 region_ids: Tensor, scale: float, chunk: int = 128):
        self.layer, self.q, self.k, self.v = layer, q, k, v               # [H, L, d], [kvh, L, d]
        self.vs, self.ve = visual
        self.P, self.L = prompt_len, q.shape[1]
        self.T = self.L - self.P                                           # decode steps simulated
        self.H, self.kvh, self.d = q.shape[0], k.shape[0], q.shape[-1]
        self.grp = self.H // self.kvh
        self.rid = region_ids.to(q.device)
        self.R = int(self.rid.max()) + 1
        self.n_vis = self.ve - self.vs
        self.scale, self.chunk = scale, chunk
        self.pooled = pool_regions(k[:, self.vs:self.ve].float(), v[:, self.vs:self.ve].float(), self.rid, self.R)
        self._not_visual = torch.ones(self.L, dtype=torch.bool, device=q.device)
        self._not_visual[self.vs:self.ve] = False
        self._pos = torch.arange(self.L, device=q.device)

    def decode_positions(self, with_prefill: bool = False) -> Tensor:
        return torch.arange(self.P - 1 if with_prefill else self.P, self.L, device=self.q.device)

    def _qg(self, pos: Tensor) -> Tensor:                                  # [kvh, grp·n, d], head h = kv·grp + i
        return self.q[:, pos].float().reshape(self.kvh, self.grp * pos.numel(), self.d)

    def _ungroup(self, x: Tensor, n: int) -> Tensor:                       # [kvh, grp·n, ...] → [H, n, ...]
        return x.reshape(self.H, n, *x.shape[2:])

    def other_lse(self, pos: Tensor) -> Tensor:
        """[H, n] logsumexp of scale·q·k over the non-visual keys at positions ≤ each query's."""
        out = []
        kf = self.k.float()
        for c in range(0, pos.numel(), self.chunk):
            p = pos[c:c + self.chunk]
            logits = self.scale * (self._qg(p) @ kf.transpose(-1, -2))
            allowed = (self._pos[None, :] <= p[:, None]) & self._not_visual[None, :]
            logits = logits.masked_fill(~allowed.repeat(self.grp, 1)[None], float("-inf"))
            out.append(self._ungroup(torch.logsumexp(logits, -1), p.numel()))
        return torch.cat(out, dim=1)

    def exact_shares(self, pos: Tensor) -> Tensor:
        """[H, n, R] true per-head log attention share of each region."""
        out = []
        kv = self.k[:, self.vs:self.ve].float()
        for c in range(0, pos.numel(), self.chunk):
            p = pos[c:c + self.chunk]
            mass = exact_region_log_mass(self._qg(p), kv, self.rid, self.R, self.scale)
            out.append(self._ungroup(mass, p.numel()))
        mass = torch.cat(out, dim=1)
        return region_log_shares(mass, self.other_lse(pos))

    def estimated_shares(self, pos: Tensor, cov_mode: str = "capped") -> Tensor:
        """[H, n, R] per-head log shares estimated from the stand-ins (what CueKeep can see)."""
        s = demand_scores(self._qg(pos), self.pooled, self.scale, cov_mode=cov_mode)
        return region_log_shares(self._ungroup(s, pos.numel()), self.other_lse(pos))


# ----------------------------------------------------------------------------- policies

def _budget_regions(R: int, frac: float) -> int:
    return max(1, min(R, round(frac * R)))


class Policy:
    standins = True
    name = "policy"

    def residency(self, ctx: LayerContext) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        """→ (tok_keep [T|1, n_vis] bool, pooled_use [T|1, R] bool, stats)."""
        raise NotImplementedError

    @staticmethod
    def _from_regions(ctx: LayerContext, resident: Tensor) -> Tuple[Tensor, Tensor]:
        return resident[:, ctx.rid], ~resident


class AllResident(Policy):
    name = "all_resident"

    def residency(self, ctx):
        dev = ctx.q.device
        return (torch.ones(1, ctx.n_vis, dtype=torch.bool, device=dev),
                torch.zeros(1, ctx.R, dtype=torch.bool, device=dev), {"exact_frac": 1.0})


class StandinsOnly(Policy):
    name = "standins_only"

    def residency(self, ctx):
        resident = torch.zeros(1, ctx.R, dtype=torch.bool, device=ctx.q.device)
        tok, use = self._from_regions(ctx, resident)
        return tok, use, {"exact_frac": 0.0, "entries": ctx.R}


class RandomRegions(Policy):
    name = "random_regions"

    def __init__(self, frac: float, seed: int):
        self.frac, self.seed = frac, seed

    def residency(self, ctx):
        C = _budget_regions(ctx.R, self.frac)
        gen = torch.Generator().manual_seed(self.seed * 1000 + ctx.layer)
        resident = torch.zeros(1, ctx.R, dtype=torch.bool)
        resident[0, torch.randperm(ctx.R, generator=gen)[:C]] = True
        tok, use = self._from_regions(ctx, resident.to(ctx.q.device))
        return tok, use, {"regions_resident": C}


class Oracle(Policy):
    name = "oracle"

    def __init__(self, frac: float):
        self.frac = frac

    def residency(self, ctx):
        C = _budget_regions(ctx.R, self.frac)
        mass = ctx.exact_shares(ctx.decode_positions()).exp().sum(0)            # [T, R]
        idx = torch.topk(mass, C, dim=-1).indices
        resident = torch.zeros_like(mass, dtype=torch.bool).scatter_(-1, idx, True)
        tok, use = self._from_regions(ctx, resident)
        return tok, use, {"regions_resident": C}


class CueKeep(Policy):
    name = "cuekeep"

    def __init__(self, frac: float, method: dict):
        self.frac, self.m = frac, method

    def residency(self, ctx):
        m = self.m
        C = _budget_regions(ctx.R, self.frac)
        shares = ctx.estimated_shares(ctx.decode_positions(with_prefill=True), m.get("cov_mode", "capped"))
        if m.get("score_agg", "max") == "max":
            agg = shares.amax(0)
        else:
            agg = torch.logsumexp(shares, 0) - torch.log(torch.tensor(float(ctx.H)))
        agg = agg.float().cpu()                                                     # [T+1, R]: one sync per layer
        pool = SlotPool(capacity=C, fetch_budget=int(m.get("fetch_budget", 2)),
                        ema_decay=float(m.get("ema_decay", 0.9)), min_residency=int(m.get("min_residency", 4)),
                        admit_margin=float(m.get("admit_margin", 0.0)))
        refresh, lookahead = int(m.get("refresh_every", 1)), int(m.get("lookahead", 1))
        fetch_budget = pool.fetch_budget
        committed, pending, fetches = set(), [], 0
        resident = torch.zeros(ctx.T, ctx.R, dtype=torch.bool)

        def plan(row: int, t: int) -> None:
            nonlocal pending, fetches
            f, ev = pool.plan(agg[row], t)
            for r, _ in ev:
                committed.discard(r)
                pending = [x for x in pending if x[0] != r]
            pending += [(r, t + max(lookahead, 1)) for r, _ in f]
            fetches += len(f)

        pool.fetch_budget = C if m.get("prefill_fill", True) else fetch_budget
        plan(0, -1)                                    # end of prefill: score with the last prompt query
        pool.fetch_budget = fetch_budget
        for t in range(ctx.T):
            still = []
            for r, ready in pending:
                (committed.add(r) if ready <= t else still.append((r, ready)))
            pending = still
            if committed:
                resident[t, list(committed)] = True
            if t % refresh == 0:
                plan(t + 1, t)                         # after step t's attention, with step t's query
        tok, use = self._from_regions(ctx, resident.to(ctx.q.device))
        return tok, use, {"regions_resident": float(resident.sum(1).float().mean()), "capacity": C,
                          "fetches_per_step": fetches / max(ctx.T, 1)}


class PruneOnce(Policy):
    name = "prune_once"
    standins = False

    def __init__(self, frac: float):
        self.frac = frac

    def residency(self, ctx):
        C = _budget_regions(ctx.R, self.frac)
        K = min(ctx.n_vis, round(self.frac * ctx.n_vis + (ctx.R - C)))           # CueKeep's entries: exact + stand-ins
        p = torch.tensor([ctx.P - 1], device=ctx.q.device)
        logits = ctx.scale * (ctx._qg(p) @ ctx.k.float().transpose(-1, -2))       # [kvh, grp, L]
        logits = logits.masked_fill((ctx._pos > ctx.P - 1)[None, None], float("-inf"))
        attn = torch.softmax(logits, -1)[..., ctx.vs:ctx.ve].sum((0, 1))          # [n_vis], heads summed
        keep = torch.zeros(1, ctx.n_vis, dtype=torch.bool, device=ctx.q.device)
        keep[0, torch.topk(attn, K).indices] = True
        return keep, torch.zeros(1, ctx.R, dtype=torch.bool, device=ctx.q.device), {"tokens_kept": K}


# ----------------------------------------------------------------------------- attention

def paged_attention(ctx: LayerContext, tok_keep: Tensor, pooled_use: Tensor, standins: bool) -> Tensor:
    """Attention output [H, T, d] (float32) of the decode queries under the given residency."""
    kf = torch.cat([ctx.k.float(), ctx.pooled.keys.float()], dim=1)               # [kvh, L+R, d]
    vf = torch.cat([ctx.v.float(), ctx.pooled.values.float()], dim=1)
    log_m = ctx.pooled.log_m.float().to(kf.device)
    pos = ctx.decode_positions()
    out = []
    for c in range(0, pos.numel(), ctx.chunk):
        p = pos[c:c + ctx.chunk]
        t = p - ctx.P
        n = p.numel()
        mask = torch.zeros(n, ctx.L + ctx.R, device=kf.device)
        mask[:, :ctx.L] = torch.where(ctx._pos[None, :] <= p[:, None], 0.0, float("-inf"))
        keep = tok_keep[t] if tok_keep.shape[0] > 1 else tok_keep.expand(n, -1)
        mask[:, ctx.vs:ctx.ve] = mask[:, ctx.vs:ctx.ve].masked_fill(~keep, float("-inf"))
        use = pooled_use[t] if pooled_use.shape[0] > 1 else pooled_use.expand(n, -1)
        if standins:
            mask[:, ctx.L:] = torch.where(use, log_m[None, :], torch.tensor(float("-inf"), device=kf.device))
        else:
            mask[:, ctx.L:] = float("-inf")
        logits = ctx.scale * (ctx._qg(p) @ kf.transpose(-1, -2)) + mask.repeat(ctx.grp, 1)[None]
        out.append(ctx._ungroup(torch.softmax(logits, -1) @ vf, n))
    return torch.cat(out, dim=1)


class PagingSimulation:
    """Context manager: during a teacher-forced forward over prompt + trace, replace the decode-position
    attention of `layers` with `paged_attention` under `policy`. Prompt positions are untouched."""

    def __init__(self, model, layers, visual: Tuple[int, int], prompt_len: int, region_ids: Tensor,
                 policy: Policy, chunk: int = 128):
        mods = _text_attention_modules(model)
        missing = [l for l in layers if l not in mods]
        if missing:
            raise ValueError(f"layers {missing} not in the model (it has {len(mods)} text layers)")
        self.targets = {id(mods[l]): l for l in layers}
        self.impl = next(iter(mods.values())).config._attn_implementation or "sdpa"
        self.visual, self.P, self.region_ids, self.policy, self.chunk = visual, prompt_len, region_ids, policy, chunk
        self.stats: Dict[int, Dict[str, float]] = {}

    def __enter__(self) -> "PagingSimulation":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self.impl == "eager" or self.impl not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(f"PagingSimulation needs a registered attention kernel such as 'sdpa', got {self.impl!r}")
        self._registry = ALL_ATTENTION_FUNCTIONS
        self._had_local = self.impl in ALL_ATTENTION_FUNCTIONS._local_mapping
        self._prev = ALL_ATTENTION_FUNCTIONS._local_mapping.get(self.impl)
        base = ALL_ATTENTION_FUNCTIONS[self.impl]

        def wrapped(module, query, key, value, attention_mask, **kwargs):
            out, weights = base(module, query, key, value, attention_mask, **kwargs)
            layer = self.targets.get(id(module))
            if layer is not None and query.shape[2] > self.P:
                self._apply(layer, module, query, key, value, out)
            return out, weights

        ALL_ATTENTION_FUNCTIONS[self.impl] = wrapped
        return self

    def __exit__(self, *exc) -> None:
        if self._had_local:
            self._registry[self.impl] = self._prev
        else:
            del self._registry[self.impl]

    @torch.no_grad()
    def _apply(self, layer, module, query, key, value, out) -> None:
        if query.shape[0] != 1:
            raise ValueError("PagingSimulation supports batch size 1")
        ctx = LayerContext(layer, query[0], key[0], value[0], self.visual, self.P, self.region_ids,
                           float(getattr(module, "scaling", query.shape[-1] ** -0.5)), self.chunk)
        tok_keep, pooled_use, stats = self.policy.residency(ctx)
        att = paged_attention(ctx, tok_keep, pooled_use, self.policy.standins)          # [H, T, d]
        out[0, ctx.P:] = att.transpose(0, 1).to(out.dtype)
        self.stats[layer] = stats


# ----------------------------------------------------------------------------- comparison

@torch.no_grad()
def compare_next_token(h_ref: Tensor, h: Tensor, lm_head, chunk: int = 256) -> Dict[str, Tensor]:
    """Per-position KL(ref ‖ h) and top-1 agreement of the next-token distributions. h_*: [T, hidden]."""
    kls, agree = [], []
    for c in range(0, h_ref.shape[0], chunk):
        lp_ref = torch.log_softmax(lm_head(h_ref[c:c + chunk]).float(), -1)
        lp = torch.log_softmax(lm_head(h[c:c + chunk]).float(), -1)
        kls.append((lp_ref.exp() * (lp_ref - lp)).sum(-1).clamp_min(0).cpu())
        agree.append((lp_ref.argmax(-1) == lp.argmax(-1)).cpu())
    return {"kl": torch.cat(kls), "top1": torch.cat(agree)}

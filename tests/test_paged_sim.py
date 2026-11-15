"""The masked paging simulation: its attention must equal the reference math for every residency."""
import pytest

torch = pytest.importorskip("torch")

from cuekeep.cache import attend_mixed, pool_regions, region_ids_for_grid  # noqa: E402
from cuekeep.sim.paged_sim import (  # noqa: E402
    AllResident, CueKeep, LayerContext, Oracle, PruneOnce, RandomRegions, StandinsOnly, paged_attention,
)


def _ctx(seed=0, h=4, w=6, g=2, H=4, kvh=2, d=8, n_text_before=5, P_extra=3, T=12, chunk=5):
    gen = torch.Generator().manual_seed(seed)
    n_vis = h * w
    vs = n_text_before
    ve = vs + n_vis
    P = ve + P_extra
    L = P + T
    q = torch.randn(H, L, d, generator=gen)
    k = torch.randn(kvh, L, d, generator=gen)
    v = torch.randn(kvh, L, d, generator=gen)
    rid = region_ids_for_grid(h, w, g)
    return LayerContext(0, q, k, v, (vs, ve), P, rid, d ** -0.5, chunk=chunk)


def _reference(ctx, resident_by_t, standins=True, tok_keep_by_t=None):
    """attend_mixed per decode position, with the non-visual keys ≤ p as dense keys."""
    out = []
    grp = ctx.H // ctx.kvh
    pooled = pool_regions(ctx.k[:, ctx.vs:ctx.ve], ctx.v[:, ctx.vs:ctx.ve], ctx.rid, ctx.R)
    for t in range(ctx.T):
        p = ctx.P + t
        dense = [j for j in range(p + 1) if not (ctx.vs <= j < ctx.ve)]
        qp = ctx.q[:, p].reshape(ctx.kvh, grp, ctx.d)
        if tok_keep_by_t is None:
            o = attend_mixed(qp, ctx.k[:, dense], ctx.v[:, dense], ctx.k[:, ctx.vs:ctx.ve], ctx.v[:, ctx.vs:ctx.ve],
                             ctx.rid, pooled, resident_by_t[t], ctx.scale, standins=standins)
        else:                                   # token-level keep, no stand-ins (prune-once)
            keep = [ctx.vs + i for i in range(ctx.n_vis) if tok_keep_by_t[t][i]]
            idx = dense + keep
            logits = ctx.scale * qp @ ctx.k[:, idx].transpose(-1, -2)
            o = torch.softmax(logits, -1) @ ctx.v[:, idx]
        out.append(o.reshape(ctx.H, ctx.d))
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paged_attention_matches_reference_for_region_policies(seed):
    ctx = _ctx(seed)
    for policy in (AllResident(), StandinsOnly(), RandomRegions(0.34, seed), Oracle(0.34),
                   CueKeep(0.34, {"fetch_budget": 1, "min_residency": 1, "refresh_every": 1, "lookahead": 1})):
        tok_keep, use, _ = policy.residency(ctx)
        got = paged_attention(ctx, tok_keep, use, policy.standins)
        resident = (~use).expand(ctx.T, -1) if use.shape[0] == 1 else ~use
        if isinstance(policy, AllResident):
            resident = torch.ones(ctx.T, ctx.R, dtype=torch.bool)
        ref = _reference(ctx, resident)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_all_resident_equals_plain_causal_attention():
    ctx = _ctx()
    tok, use, _ = AllResident().residency(ctx)
    got = paged_attention(ctx, tok, use, True)
    grp = ctx.H // ctx.kvh
    ref = []
    for t in range(ctx.T):
        p = ctx.P + t
        qp = ctx.q[:, p].reshape(ctx.kvh, grp, ctx.d)
        ref.append((torch.softmax(ctx.scale * qp @ ctx.k[:, :p + 1].transpose(-1, -2), -1) @ ctx.v[:, :p + 1]).reshape(ctx.H, ctx.d))
    torch.testing.assert_close(got, torch.stack(ref, 1), rtol=1e-5, atol=1e-5)


def test_prune_once_keeps_matched_entries_without_standins():
    ctx = _ctx()
    pol = PruneOnce(0.34)
    tok, use, stats = pol.residency(ctx)
    C = max(1, round(0.34 * ctx.R))
    assert int(tok.sum()) == stats["tokens_kept"] == min(ctx.n_vis, round(0.34 * ctx.n_vis + ctx.R - C))
    assert not bool(use.any()) and pol.standins is False
    got = paged_attention(ctx, tok, use, pol.standins)
    torch.testing.assert_close(got, _reference(ctx, None, tok_keep_by_t=[tok[0].tolist()] * ctx.T), rtol=1e-5, atol=1e-5)


def test_cuekeep_policy_is_causal_capped_and_uses_lookahead():
    ctx = _ctx(T=20)
    m = {"fetch_budget": 1, "min_residency": 2, "refresh_every": 1, "lookahead": 1, "prefill_fill": True}
    tok, use, stats = CueKeep(0.34, m).residency(ctx)
    resident = ~use
    C = max(1, round(0.34 * ctx.R))
    assert int(resident.sum(1).max()) <= C                     # hard cap
    assert int(resident[0].sum()) == C                         # prefill fill is visible at step 0
    # Changing a later query must not change earlier residency (causality).
    ctx2 = _ctx(T=20)
    ctx2.q[:, ctx2.P + 10:] = torch.randn_like(ctx2.q[:, ctx2.P + 10:])
    _, use2, _ = CueKeep(0.34, m).residency(ctx2)
    assert torch.equal(use[:11], use2[:11])                    # step-10 query affects step 11 onwards only


def test_oracle_picks_true_top_regions_per_step():
    ctx = _ctx(seed=3)
    tok, use, _ = Oracle(0.25).residency(ctx)
    mass = ctx.exact_shares(ctx.decode_positions()).exp().sum(0)
    C = max(1, round(0.25 * ctx.R))
    assert torch.equal((~use).sum(1), torch.full((ctx.T,), C))
    top = torch.topk(mass, C, -1).indices
    assert bool((~use).gather(1, top).all())

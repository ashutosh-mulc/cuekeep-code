"""Capture what decoding attends to, from one teacher-forced pass of a Qwen3-VL model.

Decoding step s (0-based) is the forward of the token at absolute position `prompt_len - 1 + s`: its
query attends to every key at positions ≤ that one. So a single causal forward over prompt + trace
yields, for any chosen steps, exactly the queries and keys that decoding used. That is cheaper and
simpler than hooking into generation, and the integration test checks it reproduces greedy decoding.

`KVCapture` temporarily wraps the attention function that the text-attention modules dispatch to
(`ALL_ATTENTION_FUNCTIONS[<impl>]`, e.g. "sdpa"). The wrapped call computes the model's real output
unchanged; for the target layers it also records, per query set (e.g. "spread", "dense"):
  q          [heads, P, d]   queries after q_norm and RoPE, as the model computed them
  attn_out   [heads, P, d]   the model's per-head attention output at those positions
  other_lse  [heads, P]      logsumexp of scale·q·k over the non-visual keys at positions ≤ the query's
  other_out  [heads, P, d]   those keys' attention-weighted mean value
and, once per layer, the visual K/V `k`, `v` [kvh, n_vis, d] (after RoPE). Vision-tower attention and
non-target layers pass straight through. See analysis/dumps.py for how these become dump units.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor


def _text_attention_modules(model) -> Dict[int, torch.nn.Module]:
    """layer_idx -> attention module of the language model (not the vision tower)."""
    out = {}
    for name, mod in model.named_modules():
        if type(mod).__name__.endswith("TextAttention") and hasattr(mod, "layer_idx"):
            out[int(mod.layer_idx)] = mod
    if not out:
        raise RuntimeError("no text-attention modules found (expected *TextAttention with layer_idx)")
    return out


class KVCapture:
    def __init__(self, model, layers: Iterable[int], visual: Tuple[int, int],
                 query_positions: Dict[str, Tensor], chunk: int = 64, store_dtype=torch.bfloat16):
        mods = _text_attention_modules(model)
        missing = [l for l in layers if l not in mods]
        if missing:
            raise ValueError(f"layers {missing} not in the model (it has {len(mods)} text layers)")
        self.targets = {id(mods[l]): l for l in layers}
        self.impl = next(iter(mods.values())).config._attn_implementation or "sdpa"
        self.visual = visual
        self.positions = {k: torch.as_tensor(v, dtype=torch.long) for k, v in query_positions.items()}
        self.chunk = chunk
        self.store_dtype = store_dtype
        self.results: Dict[int, Dict[str, object]] = {}
        self._registry = None
        self._had_local = False
        self._prev = None

    # -- install / uninstall ---------------------------------------------------------------------
    def __enter__(self) -> "KVCapture":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self.impl == "eager" or self.impl not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(f"KVCapture needs a registered attention kernel such as 'sdpa', got {self.impl!r}")
        self._registry = ALL_ATTENTION_FUNCTIONS
        self._had_local = self.impl in ALL_ATTENTION_FUNCTIONS._local_mapping
        self._prev = ALL_ATTENTION_FUNCTIONS._local_mapping.get(self.impl)
        base = ALL_ATTENTION_FUNCTIONS[self.impl]

        def wrapped(module, query, key, value, attention_mask, **kwargs):
            out, weights = base(module, query, key, value, attention_mask, **kwargs)
            layer = self.targets.get(id(module))
            if layer is not None:
                self._record(layer, module, query, key, value, out)
            return out, weights

        ALL_ATTENTION_FUNCTIONS[self.impl] = wrapped
        return self

    def __exit__(self, *exc) -> None:
        if self._had_local:
            self._registry[self.impl] = self._prev
        else:
            del self._registry[self.impl]

    # -- recording ---------------------------------------------------------------------------------
    @torch.no_grad()
    def _record(self, layer: int, module, query: Tensor, key: Tensor, value: Tensor, out: Tensor) -> None:
        if query.shape[0] != 1:
            raise ValueError("KVCapture supports batch size 1")
        q, k, v = query[0], key[0], value[0]                        # [H, L, d], [kvh, L, d], [kvh, L, d]
        H, L, d = q.shape
        kvh = k.shape[0]
        grp = H // kvh
        if out.dim() != 4 or out.shape[1] != L or out.shape[2] != H:
            raise ValueError(f"unexpected attention output shape {tuple(out.shape)}; expected [1, {L}, {H}, {d}]")
        vs, ve = self.visual
        rec = self.results.setdefault(layer, {})
        if "k" not in rec:
            rec["k"] = k[:, vs:ve].to(self.store_dtype).cpu()
            rec["v"] = v[:, vs:ve].to(self.store_dtype).cpu()
        scale = float(getattr(module, "scaling", d ** -0.5))
        pos_all = torch.arange(L, device=q.device)
        not_visual = (pos_all < vs) | (pos_all >= ve)
        kf, vf = k.float(), v.float()
        for name, pos in self.positions.items():
            pos = pos.to(q.device)
            if int(pos.max()) >= L or int(pos.min()) < ve:
                raise ValueError(f"{name} positions must lie in [{ve}, {L}) (after the image), got [{int(pos.min())}, {int(pos.max())}]")
            lse_parts, out_parts = [], []
            for c in range(0, pos.numel(), self.chunk):
                p = pos[c:c + self.chunk]                                            # [P']
                qg = q[:, p].float().reshape(kvh, grp * p.numel(), d)                # head h = kv·grp + i
                logits = scale * (qg @ kf.transpose(-1, -2))                         # [kvh, grp·P', L]
                allowed = (pos_all[None, :] <= p[:, None]) & not_visual[None, :]     # [P', L]
                logits = logits.masked_fill(~allowed.repeat(grp, 1)[None], float("-inf"))
                lse_parts.append(torch.logsumexp(logits, dim=-1).reshape(H, p.numel()))
                out_parts.append((torch.softmax(logits, dim=-1) @ vf).reshape(H, p.numel(), d))
            rec[name] = {
                "q": q[:, pos].to(self.store_dtype).cpu(),
                "attn_out": out[0, pos].transpose(0, 1).to(self.store_dtype).cpu(),     # [H, P, d]
                "other_lse": torch.cat(lse_parts, dim=1).cpu(),
                "other_out": torch.cat(out_parts, dim=1).cpu(),
            }


def pad_like_input_ids(inputs: Dict[str, Tensor], full_ids: Tensor) -> Dict[str, Tensor]:
    """Model kwargs for a teacher-forced pass over `full_ids` (prompt + trace), from the prompt's inputs.

    input_ids is replaced, attention_mask is extended with ones, and any other per-token tensor
    (e.g. mm_token_type_ids, shape [1, prompt_len]) is extended with zeros (text tokens).
    """
    n_prompt = inputs["input_ids"].shape[1]
    extra = full_ids.shape[1] - n_prompt
    out = {}
    for name, t in inputs.items():
        if name == "input_ids":
            out[name] = full_ids
        elif isinstance(t, Tensor) and t.dim() == 2 and t.shape == (1, n_prompt):
            fill = 1 if name == "attention_mask" else 0
            out[name] = torch.cat([t, t.new_full((1, extra), fill)], dim=1)
        else:
            out[name] = t
    return out

"""Shared template-prefix key/value cache for Qwen3-0.6B on two PyTorch MPS builds (engine).

The chat template opens every prompt with the same 3 tokens, `<|im_start|>user\\n` (positions 0-2; checked in
`prefix_ids`). A prefix cache stores the keys (K, after k_norm and RoPE) and values (V) of these 3 tokens in every
attention layer, made ONCE by the grading build (eager), and both builds reuse it: each build then runs only
positions 3-33 and their queries read the 3 prefix keys/values from the cache. Positions 0-2 reach later tokens only
through K and V, so with one shared cache the prefix is bit-identical for both builds by construction.

Attention: a registered attention function ('pfx'):
  - prefix pass (no `pfx_kv` kwarg): captures each layer's K, V into `pfx_cap` and runs the stock sdpa function
    (is_causal, no mask) = the stock prefill of 3 tokens.
  - rest pass (`pfx_kv` = tuple of (K, V) per layer): concatenates the cached prefix K/V in front of the rest's K/V and
    runs the stock sdpa function with an explicit causal mask [T_rest, 3 + T_rest] (query i at position 3+i sees keys
    0..3+i). RoPE positions of the rest are 3..33 (position_ids).
'pfx' is not in the mask registry, so transformers builds no mask of its own (masking_utils: custom attention).
"""
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

P = 3  # template prefix length


def pfx_attention(module, query, key, value, attention_mask, pfx_kv=None, pfx_cap=None, **kw):
    li = module.layer_idx
    if pfx_kv is None:
        if pfx_cap is not None:
            pfx_cap.append((key, value))
        return sdpa_attention_forward(module, query, key, value, None, **kw)
    pk, pv = pfx_kv[li]
    b = key.shape[0]
    k = torch.cat([pk.expand(b, -1, -1, -1), key], 2)
    v = torch.cat([pv.expand(b, -1, -1, -1), value], 2)
    t = query.shape[2]
    mask = torch.ones(t, P + t, dtype=torch.bool, device=query.device).tril(diagonal=P)[None, None]
    return sdpa_attention_forward(module, query, k, v, mask, **kw)


AttentionInterface.register("pfx", pfx_attention)


def use_pfx(model) -> None:
    model.config._attn_implementation = "pfx"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "pfx"


def _prefix_fwd(model, prefix_ids, tail):
    cap: list = []
    out = model.model(input_ids=prefix_ids, use_cache=False, output_hidden_states=True, pfx_cap=cap)
    return tuple(cap), out.hidden_states[tail][0]


@torch.no_grad()
def prefix_pass(model, prefix_ids: torch.Tensor, tail: int, build: str = "eager", full_ids: torch.Tensor | None = None):
    """Prefill of the 3 template tokens on `build` (eager = the grading build, the shared cache; compiled/emu = a
    build's OWN cache): per-layer (K, V) [1, kv_heads, 3, d] and the block-`tail` input rows [3, H].
    eager_whole = K/V at positions 0-2 taken from ONE whole 34-token eager pass (`full_ids`): the exact prefix bits the
    stock grading graph used in stack-conditional (positions 0-2 are identical across prompts within a build)."""
    import torch._inductor.config as icfg
    if build == "eager":
        cap, rows = _prefix_fwd(model, prefix_ids, tail)
    elif build == "eager_whole":
        assert full_ids is not None and torch.equal(full_ids[:, :P], prefix_ids)
        cap, rows = _prefix_fwd(model, full_ids, tail)
        cap = tuple((k[:, :, :P], v[:, :, :P]) for k, v in cap)
        rows = rows[:P]
    else:
        _prefix_fwd(model, prefix_ids, tail)  # eager warm-up: installs the hidden-state hooks outside the graph
        old = icfg.emulate_precision_casts
        icfg.emulate_precision_casts = build == "emu"
        torch._dynamo.reset()
        cap, rows = torch.compile(_prefix_fwd, fullgraph=True)(model, prefix_ids, tail)
        torch._dynamo.reset()
        icfg.emulate_precision_casts = old
    assert len(cap) == model.config.num_hidden_layers, len(cap)
    return tuple((k.clone(), v.clone()) for k, v in cap), rows.clone()


def rest_positions(t_rest: int, device) -> torch.Tensor:
    return torch.arange(P, P + t_rest, device=device)[None]

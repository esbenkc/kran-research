"""K / V patch engine for the cached 2-block tail (blocks 26-27 of Qwen3-0.6B).

A "patch" copies the condition-off run's keys and/or values (after k_norm and RoPE, i.e. exactly what the attention
uses) at chosen key positions into the condition-on run, for chosen QUERY heads only (GQA: 16 query heads share 8
KV heads per block; the donor K/V is repeated to 16 heads first, so a patched head never leaks into its sibling).
The patch acts on every query position >= 3 in both blocks; queries at positions 0-2 run unpatched, so the
condition-on run's own tokens 0-2 (and its block-27 K/V there) are untouched. Tokens 0-2 can reach the answer only
through their K and V in blocks 26-27, so donor K+V at 0-2 for all heads == massive-dims row (d) (gate G0).

Donor K/V come from one capture pass of the SAME copy on the condition-off input.
share moved = (P_patched - P_on) / (P_off - P_on), P = mean first-token probability mass on the X tokens.

Arm spec (per block, 0 = block 26, 1 = block 27): dict(heads=[q-head ids], kpos=[...], vpos=[...], vmode=...)
vmode: "donor" (copy donor V), "mirror" (2 V_on - V_off), "rand" (V_on + random direction with the norm of
V_off - V_on, per prompt / head / position; seeded). qfrom: first query position that gets the patch (default 3;
R7 uses the answer position only). hold0 (R8): the patched heads' attention weight on token 0 is held at the
recipient's value and the weights on the other keys are rescaled to fill the rest (fp32 math, manual softmax).
manual: compute the patched heads with the same manual fp32 softmax as hold0 (the R8 comparison row).
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

CTRL = {"mode": "none", "spec": None, "donor": None, "rng": None, "attn": None}
NQ = 16
FIRST = 3          # first query position that gets the patch
CAP = {}


def _attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    li = module.layer_idx - 26
    mode = CTRL["mode"]
    if mode == "capture":
        CAP[li] = (key.detach().clone(), value.detach().clone())
        CAP[f"q{li}"] = query.detach().clone()
    if mode == "attn":   # record attention weights (fp32 math) from every query position to every key
        n_rep = module.num_key_value_groups
        k = repeat_kv(key, n_rep).float()
        sc = (query.float() @ k.transpose(-1, -2)) * scaling
        T_ = sc.shape[-1]
        sc = sc.masked_fill(torch.ones(T_, T_, dtype=torch.bool, device=sc.device).triu(1), float("-inf"))
        CTRL["attn"][li].append(torch.softmax(sc, -1).cpu())
    spec = CTRL["spec"][li] if (mode == "patch" and CTRL["spec"] is not None) else None
    out_r, _ = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kw)
    if spec is None or not spec.get("heads"):
        return out_r, None
    n_rep = module.num_key_value_groups
    kd, vd = (repeat_kv(t, n_rep) for t in CTRL["donor"][li])
    k = repeat_kv(key, n_rep).clone()
    v = repeat_kv(value, n_rep).clone()
    H = torch.tensor(spec["heads"], device=k.device)
    for p in spec.get("kpos", []):
        k[:, H, p] = kd[:, H, p]
    vmode = spec.get("vmode", "donor")
    for p in spec.get("vpos", []):
        if vmode == "donor":
            v[:, H, p] = vd[:, H, p]
        elif vmode == "mirror":
            v[:, H, p] = (2 * v[:, H, p].float() - vd[:, H, p].float()).to(v.dtype)
        elif vmode == "rand":
            von, voff = v[:, H, p].float(), vd[:, H, p].float()
            nrm = (voff - von).norm(dim=-1, keepdim=True)
            g = torch.from_numpy(CTRL["rng"].standard_normal(von.shape).astype(np.float32)).to(von.device)
            g = g / g.norm(dim=-1, keepdim=True)
            v[:, H, p] = (von + nrm * g).to(v.dtype)
        else:
            raise ValueError(vmode)
    qf = spec.get("qfrom", FIRST)
    if spec.get("hold0") or spec.get("manual"):
        kr = repeat_kv(key, n_rep).float()
        q32 = query.float()
        T_ = q32.shape[2]
        causal = torch.ones(T_, T_, dtype=torch.bool, device=q32.device).triu(1)
        a_pat = torch.softmax(((q32 @ k.float().transpose(-1, -2)) * scaling).masked_fill(causal, float("-inf")), -1)
        if spec.get("hold0"):
            a_rec = torch.softmax(((q32 @ kr.transpose(-1, -2)) * scaling).masked_fill(causal, float("-inf")), -1)
            a = a_pat.clone()
            a[..., 0] = a_rec[..., 0]
            a[..., 1:] = a_pat[..., 1:] * ((1 - a_rec[..., :1]) / (1 - a_pat[..., :1]))
        else:
            a = a_pat
        out_p = (a @ v.float()).to(query.dtype).transpose(1, 2)
    else:
        out_p = F.scaled_dot_product_attention(query, k, v, scale=scaling, is_causal=True).transpose(1, 2)
    out = out_r.clone()
    out[:, qf:, H] = out_p[:, qf:, H]      # only the patched heads change; every other head stays bit-identical
    return out.contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("kvpatch", _attn)


def install(tail):
    for layer in tail.layers:
        layer.self_attn.config._attn_implementation = "kvpatch"
    assert [l.self_attn.layer_idx for l in tail.layers] == [26, 27]


def arm(heads26, heads27, kpos=(), vpos=(), **kw):
    return [dict(heads=list(heads26), kpos=list(kpos), vpos=list(vpos), **kw),
            dict(heads=list(heads27), kpos=list(kpos), vpos=list(vpos), **kw)]

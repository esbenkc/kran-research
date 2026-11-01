"""R10 freeze (Iris, PREREG row R10): a FIRST-ORDER prediction of the K-only and V-only patch shares, per query head, before any
patch runs. Attribution patching (Nanda 2023; Syed et al. arXiv 2310.10348): for each query head h of blocks 26-27
and each token group g in {token 0, tokens 1-2}, scale the donor-minus-recipient difference of K (after k_norm and
RoPE) or V by alpha[h,g] and take dP(X)/dalpha at alpha = 0 on the recipient (cue-on) run. First-order effects add,
so the all-heads V-only prediction is the sum over heads. share = dP / (P_off - P_on) (whole-switch denominator).

Also (reading rows, no patch): per-head attention on tokens 0-2 on the BASE tail (query positions >= 3 and the answer
token), the sink set (mean attention on 0-2 over query positions >= 3 > 0.5), and the value-norm ratio
|W_O,h v_0| / mean_j>=3 |W_O,h v_j| (Guo 2410.13835 'value-state drain'; Fesser 2606.08105 nop-sink diagnostic).

Code check: our own forward equals the HF tail (alpha = 0), and the exact K+V patch at alpha = 1 (all heads, both
groups) equals the block-26-input patch of tokens 0-2 (massive-dims row d), in fp32 on CPU.

    env/.venv/bin/python kran-research/projects/sink-value/research_checks/first_order.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["NC_DEV"] = "cpu"
MD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MD))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402

import common as C  # noqa: E402

T = C.T
torch.set_num_threads(3)
OUT = Path(__file__).parent / "first_order.json"
NH, NKV, HD = 16, 8, 128
REP = NH // NKV

cfg, tail, w_out = T.load_tail()
tail.float()
w_out = w_out.float()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
xs = torch.tensor(T.x_ids())


def f32(u16):
    return torch.from_numpy((np.asarray(u16).astype(np.uint32) << 16).view(np.float32).copy())


def layer_fwd(layer, x, cs, don=None, aK=None, aV=None, rec=None):
    """One Qwen3 decoder layer, own attention. don = (k, v) of the donor at this layer ([B,16,T,128], post-RoPE,
    expanded to query heads). aK, aV: [16, 2] scales for token 0 / tokens 1-2. rec: dict to store k, v, attn."""
    at = layer.self_attn
    h = layer.input_layernorm(x)
    B, L, _ = h.shape
    q = at.q_norm(at.q_proj(h).view(B, L, NH, HD)).transpose(1, 2)
    k = at.k_norm(at.k_proj(h).view(B, L, NKV, HD)).transpose(1, 2)
    v = at.v_proj(h).view(B, L, NKV, HD).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *cs)
    k = k.repeat_interleave(REP, dim=1)
    v = v.repeat_interleave(REP, dim=1)
    if rec is not None:
        rec["k"], rec["v"] = k.detach(), v.detach()
    if don is not None:
        g = torch.zeros(2, L)
        g[0, 0] = 1.0
        g[1, 1:3] = 1.0
        sK = (aK @ g)[None, :, :, None]   # [1,16,L,1]
        sV = (aV @ g)[None, :, :, None]
        k = k + sK * (don[0] - k)
        v = v + sV * (don[1] - v)
    s = (q @ k.transpose(-1, -2)) * at.scaling
    s = s.masked_fill(torch.ones(L, L, dtype=torch.bool).triu(1), float("-inf"))
    a = torch.softmax(s, -1)
    if rec is not None:
        rec["a"] = a.detach()
    o = (a @ v).transpose(1, 2).reshape(B, L, NH * HD)
    x = x + at.o_proj(o)
    return x + layer.mlp(layer.post_attention_layernorm(x))


def run(x, don=None, alphas=None, recs=None):
    pos = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
    cs = tail.rotary(x, pos)
    for li, layer in enumerate(tail.layers):
        d = None if don is None else don[li]
        aK, aV = (None, None) if alphas is None else (alphas[li, 0], alphas[li, 1])
        x = layer_fwd(layer, x, cs, d, aK, aV, None if recs is None else recs[li])
    lg = tail.norm(x)[:, -1] @ w_out.T
    return torch.softmax(lg, -1)[:, xs].sum(-1)


def value_ratio(layer_idx, rec):
    """|W_O,h v_0| / mean over j>=3 |W_O,h v_j|, per query head, mean over prompts."""
    Wo = tail.layers[layer_idx].self_attn.o_proj.weight  # [1024, 2048]
    v = rec["v"]  # [B,16,L,128]
    r = []
    for h in range(NH):
        out = v[:, h] @ Wo[:, h * HD:(h + 1) * HD].T   # [B,L,1024]
        n = out.norm(dim=-1)
        r.append(float((n[:, 0] / n[:, 3:].mean(-1)).mean().detach()))
    return r


res = {"method": __doc__.split("\n\n")[0], "copies": {}, "base": {}}
t_all = time.time()

# ---------- base reading rows: sink set and value ratios (cue-on inputs; cue-off as a check) ----------
tail.load_state_dict(base_sd)
for sw in ("grader", "monitor"):
    X = C.load_inputs(sw)
    for cond in ("on", "off"):
        recs = [{}, {}]
        with torch.no_grad():
            run(f32(X[cond]), recs=recs)
        ent = {}
        for li in range(2):
            a = recs[li]["a"]  # [B,16,L,L]
            m03 = a[:, :, 3:, :3].sum(-1).mean((0, 2))        # mean over query positions >= 3
            ans = a[:, :, -1, :]
            ent[f"block{26 + li}"] = {
                "att_first3_qpos3plus": [round(float(z), 4) for z in m03],
                "att_tok0_answer": [round(float(z), 4) for z in ans[..., 0].mean(0)],
                "att_tok12_answer": [round(float(z), 5) for z in ans[..., 1:3].sum(-1).mean(0)],
                "sink_heads": [h for h in range(NH) if m03[h] > 0.5],
                "value_ratio_tok0": [round(z, 4) for z in value_ratio(li, recs[li])],
            }
        res["base"][f"{sw}_{cond}"] = ent
    print(sw, {b: len(e["sink_heads"]) for b, e in res["base"][f"{sw}_on"].items()}, flush=True)

# ---------- per copy: code checks + first-order shares ----------
for name in C.HEADLINE:
    t0 = time.time()
    sw, path = C.COPIES[name]
    tail.load_state_dict(base_sd)
    miss, unexp = tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    assert not unexp and all("rotary" in m for m in miss), (miss, unexp)
    tail.float()
    X = C.load_inputs(sw)
    on, off = f32(X["on"]), f32(X["off"])
    with torch.no_grad():
        hf_on = torch.softmax(tail(on, w_out), -1)[:, xs].sum(-1)
        drecs = [{}, {}]
        p_off = run(off, recs=drecs)
        p_on = run(on)
        don = [(r["k"], r["v"]) for r in drecs]
        # code check 1: own forward == HF tail
        chk_hf = float((p_on - hf_on).abs().max())
        # code check 2: exact K+V at alpha=1 == block-26-input patch of tokens 0-2 (row d)
        ones = torch.ones(2, 2, NH, 2)
        p_kv = run(on, don, ones)
        pd_in = on.clone()
        pd_in[:, :3] = off[:, :3]
        p_d = run(pd_in)
    Pon, Poff = float(p_on.mean()), float(p_off.mean())
    gap = Poff - Pon
    alphas = torch.zeros(2, 2, NH, 2, requires_grad=True)  # [block, K/V, head, group]
    p = run(on, don, alphas)
    p.mean().backward()
    g = (alphas.grad / gap).numpy()   # first-order share per block, path, head, group
    # same in log-odds space (mean per-prompt log-odds of X); the switch is far more linear there
    lo = lambda z: torch.log(z / (1 - z))
    gap_lo = float(lo(p_off).mean() - lo(p_on).mean())
    alphas2 = torch.zeros(2, 2, NH, 2, requires_grad=True)
    lo(run(on, don, alphas2)).mean().backward()
    g_lo = (alphas2.grad / gap_lo).numpy()
    sink = {li: res["base"][f"{sw}_on"][f"block{26 + li}"]["sink_heads"] for li in range(2)}

    def tot(path_i, heads=None, groups=(0, 1), g=g):
        s = 0.0
        for li in range(2):
            hs = range(NH) if heads is None else heads[li]
            s += sum(float(g[li, path_i, h, gg]) for h in hs for gg in groups)
        return s

    ent = {
        "switch": sw, "P_on": Pon, "P_off": Poff,
        "check_own_vs_hf_maxabs": chk_hf,
        "check_KV_alpha1_share": (float(p_kv.mean()) - Pon) / gap,
        "check_rowd_input_patch_share": (float(p_d.mean()) - Pon) / gap,
        "check_KV_vs_rowd_maxabs": float((p_kv - p_d).abs().max()),
        "pred_V_all": tot(1), "pred_K_all": tot(0),
        "pred_V_tok0": tot(1, groups=(0,)), "pred_V_tok12": tot(1, groups=(1,)),
        "pred_K_tok0": tot(0, groups=(0,)), "pred_K_tok12": tot(0, groups=(1,)),
        "pred_V_sink": tot(1, sink), "pred_K_sink": tot(0, sink),
        "logodds_gap": gap_lo,
        "lo_pred_V_all": tot(1, g=g_lo), "lo_pred_K_all": tot(0, g=g_lo),
        "lo_pred_V_tok0": tot(1, groups=(0,), g=g_lo), "lo_pred_V_tok12": tot(1, groups=(1,), g=g_lo),
        "lo_pred_K_tok0": tot(0, groups=(0,), g=g_lo), "lo_pred_K_tok12": tot(0, groups=(1,), g=g_lo),
        "lo_pred_V_sink": tot(1, sink, g=g_lo), "lo_pred_K_sink": tot(0, sink, g=g_lo),
        "per_head_share": {f"block{26 + li}": {"K_tok0": g[li, 0, :, 0].round(4).tolist(),
                                                "K_tok12": g[li, 0, :, 1].round(4).tolist(),
                                                "V_tok0": g[li, 1, :, 0].round(4).tolist(),
                                                "V_tok12": g[li, 1, :, 1].round(4).tolist()} for li in range(2)},
        "per_head_share_logodds": {f"block{26 + li}": {"K_tok0": g_lo[li, 0, :, 0].round(4).tolist(),
                                                "K_tok12": g_lo[li, 0, :, 1].round(4).tolist(),
                                                "V_tok0": g_lo[li, 1, :, 0].round(4).tolist(),
                                                "V_tok12": g_lo[li, 1, :, 1].round(4).tolist()} for li in range(2)},
        "secs": round(time.time() - t0, 1),
    }
    res["copies"][name] = ent
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in ent.items() if not k.startswith("per_head")},
          flush=True)

res["secs_total"] = round(time.time() - t_all, 1)
OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)

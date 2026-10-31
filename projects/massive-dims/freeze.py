"""Freeze, before any patch (PREREG s2): the massive set, the sink positions, and R1 / R1-in / R1-pre per switch.
Uses only the untrained base: block-26 input caches and the base layer-26/27 weights. Writes data/frozen.json.

    NC_DEV=cpu env/.venv/bin/python kran-research/projects/massive-dims/freeze.py
"""
import json
import os

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402

import common as C  # noqa: E402

T = C.T
FIRST3 = [0, 1, 2]
out = {"rule": "dim d is massive if at any of tokens 0-2 its median |x| over the 200 test prompts > 100 x the median "
               "|x| over the 1,024 dims at that token (untrained block-26 input, grading-build = eager cache)",
       "sets_per_cache": {}, "ratios_top5": {}}

# 1. massive set, on every base cache (the rule names the eager cache; the others are a check)
X = {sw: C.load_inputs(sw) for sw in C.SWITCH}
caches = {"grader_on(eager)": X["grader"]["on"], "grader_off(compiled)": X["grader"]["off"],
          "monitor_on(ref)": X["monitor"]["on"], "monitor_off(none)": X["monitor"]["off"]}
for name, u in caches.items():
    a = np.abs(C.to_f32(u[:, :3]))                       # [200, 3, 1024]
    med_d = np.median(a, axis=0)                        # per token x dim, over prompts
    med_tok = np.median(med_d, axis=1, keepdims=True)   # per token, over dims
    ratio = med_d / med_tok
    s = sorted({int(d) for t in range(3) for d in np.where(ratio[t] > 100)[0]})
    out["sets_per_cache"][name] = s
    out["ratios_top5"][name] = {f"token{t}": [[int(d), round(float(ratio[t, d]), 1), float(med_d[t, d])]
                                             for d in np.argsort(-ratio[t])[:5]] for t in range(3)}
    # tokens 0-2 depend only on the fixed template tokens (causal), so they are one vector per cache
    out.setdefault("first3_constant", {})[name] = bool((u[:, :3] == u[:1, :3]).all())
M = out["sets_per_cache"]["grader_on(eager)"]
out["massive_set"] = M
assert M == [1, 13, 35], M
nonM = [d for d in range(1024) if d not in M]

# 2. R1 / R1-in / R1-pre per switch, headline direction donor = off -> recipient = on (PREREG v1.2 s2)
cfg, tail, w_out = T.load_tail()
w_norm = tail.layers[0].input_layernorm.weight.detach().float().cpu().numpy()   # layer 26 input_layernorm (base; LoRA leaves it)
eps = cfg.rms_norm_eps


def post(x):  # x [..., 1024] fp32 -> what the tail reads after block 26's input RMSNorm
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w_norm


out["R1"] = {}
for sw in C.SWITCH:
    on, off = C.to_f32(X[sw]["on"][0, :3]).astype(np.float64), C.to_f32(X[sw]["off"][0, :3]).astype(np.float64)
    pat = on.copy(); pat[:, M] = off[:, M]
    d_pre = off - on; d_post = post(off) - post(on); d_after = post(off) - post(pat)
    e_post = (d_post ** 2).sum()
    tok = {}
    for t in range(3):
        et = (d_post[t] ** 2).sum()
        tok[f"token{t}"] = {"R1": None if et == 0 else round(float(1 - (d_after[t] ** 2).sum() / et), 4),
                            "post_energy_share_of_first3": round(float(et / e_post), 4)}
    out["R1"][sw] = {
        "R1": round(float(1 - (d_after ** 2).sum() / e_post), 4),
        "R1_in": round(float((d_post[:, M] ** 2).sum() / e_post), 4),
        "R1_pre": round(float((d_pre[:, M] ** 2).sum() / (d_pre ** 2).sum()), 4),
        "per_token": tok,
        "elements_differing_first3": [round(float((X[sw]["on"][0, t] != X[sw]["off"][0, t]).mean()), 4) for t in range(3)],
        "massive_values_on_off_token0": {int(d): [float(on[0, d]), float(off[0, d])] for d in M},
    }

# 3. sink positions: attention from the answer token (last position) to each position, per head of blocks 26-27 of
# the untrained tail, averaged over the 200 test prompts and heads; on the condition-on input of each switch.
CAP = {}


def grab(i):
    def pre(mod, args, kwargs):
        CAP[i] = (kwargs.get("hidden_states", args[0] if args else None), kwargs["position_embeddings"])
    return pre


hs = [tail.layers[i].self_attn.register_forward_pre_hook(grab(i), with_kwargs=True) for i in range(2)]
out["sinks"] = {}
for sw in C.SWITCH:
    acc = np.zeros((2, 34))
    u = X[sw]["on"]
    for s in range(0, len(u), 25):
        with torch.no_grad():
            tail(T.bf16(u[s:s + 25]).to(T.DEV), w_out)
            for i in range(2):
                h, (cos, sin) = CAP[i]
                at = tail.layers[i].self_attn
                shp = (*h.shape[:-1], -1, at.head_dim)
                q = at.q_norm(at.q_proj(h).view(shp)).transpose(1, 2)
                k = at.k_norm(at.k_proj(h).view(shp)).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                sc = (q[:, :, -1:].float() @ k.float().transpose(-1, -2)) * at.scaling   # last query sees all keys
                acc[i] += torch.softmax(sc, -1)[:, :, 0].mean(1).sum(0).cpu().numpy()
    acc /= len(u)
    mean = acc.mean(0)
    out["sinks"][sw] = {"mass_per_position_blocks26_27_mean": [round(float(v), 4) for v in mean],
                        "per_block": {f"block{26 + i}": [round(float(v), 4) for v in acc[i]] for i in range(2)},
                        "top3_positions": [int(p) for p in np.argsort(-mean)[:3]],
                        "mass_on_first3": round(float(mean[:3].sum()), 4)}
for h in hs:
    h.remove()
out["iris_crosscheck_v1.1"] = "research_checks/r1_freeze_v1.1.json: grader R1 -0.037, monitor R1 0.005"
json.dump(out, open(C.HERE / "data/frozen.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("massive_set", "sets_per_cache", "first3_constant")}))
print(json.dumps(out["R1"], indent=0)[:1500])
print({sw: (v["top3_positions"], v["mass_on_first3"]) for sw, v in out["sinks"].items()})

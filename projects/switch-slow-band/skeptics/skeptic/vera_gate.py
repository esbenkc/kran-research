"""Engine gate + capture-only screen (no band patch). Usage: python vera_gate.py SEED DT DEV
Gates: rotate_half pair map vs the model's own RoPE; noop patch = plain; all-64-pair patch at pos 0-2 = K-only t12
(sink-value: 0.984-0.986 fp32). Screen: exact per-pair score change of head 27.8 at the answer row (tokens 1, 2),
mean over prompts, and its band shares. Writes vera_gate_SEED_DT.json.
"""
import sys, json, time
import numpy as np, torch
sys.path.insert(0, "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic")
import vera_pe as V

seed, dt, dev = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
E = V.Eng(seed, dt, dev)
out = {"seed": seed, "dt": dt, "dev": dev, "theta": E.theta}

# 1) pair map: apply the model's RoPE to e_d at position p; dims d and d+64 must rotate together at angle p*inv_freq[d]
rot = E.model.model.rotary_emb
x = torch.zeros(1, 1, 5, 128, dtype=torch.float32)
pos = torch.arange(5)[None]
cos, sin = rot(x, pos)
cos, sin = cos.float(), sin.float()
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
ok = True
for d in (0, 5, 61, 62, 63):
    e = torch.zeros(1, 1, 5, 128); e[..., d] = 1.0
    r, _ = apply_rotary_pos_emb(e, e, cos, sin)
    nz = set(torch.nonzero(r[0, 0, 4].abs() > 1e-12).flatten().tolist())
    ang = float(torch.atan2(r[0, 0, 4, d + 64], r[0, 0, 4, d]))
    exp = 4 * E.theta ** (-2 * d / 128)
    ok &= nz <= {d, d + 64} and abs(ang - exp) < 1e-5 * max(1, exp)
    out[f"rope_d{d}"] = {"nonzero": sorted(nz), "angle_pos4": ang, "expected": exp}
out["pair_map_ok"] = bool(ok)

H = E.CON; HOFF = E.COFF
p_on = E.px(H); p_off = E.px(HOFF)
don = E.donor()
ALL = V.pair_dims(range(64)); NONE = torch.zeros(128, dtype=torch.bool)
p_noop = E.px(H, mode="patch", blocks=[0, 1], donor=don, dims=NONE, pos=[0, 1, 2])
p_all = E.px(H, mode="patch", blocks=[0, 1], donor=don, dims=ALL, pos=[0, 1, 2])
gap = p_off.mean() - p_on.mean()
out.update(P_on=float(p_on.mean()), P_off=float(p_off.mean()), noop_maxabs=float(np.abs(p_noop - p_on).max()),
           share_all64=float((p_all.mean() - p_on.mean()) / gap))

# 2) capture-only screen: head 27.8 (block index 1, query head 8), answer row
cap_on, rec_on = E.capture(H)
cap_off, rec_off = E.capture(HOFF)
q = cap_on[1][0][:, 8, -1].float()            # [N,128] cue-on answer query
k_on = cap_on[1][1][:, 8, :3].float()         # [N,3,128]
k_off = don[1][0, 8, :3].float()              # [3,128] donor keys (one fixed vector)
dk = k_off[None] - k_on                       # [N,3,128]
sc = 128 ** -0.5
term = (q[:, None, :] * dk) * sc              # [N,3,128]
pp = term[..., :64] + term[..., 64:]          # [N,3,64] per-pair score change at tokens 0,1,2
tot = pp.sum(-1)
out["dk_tok0_maxabs"] = float(dk[:, 0].abs().max())
out["score_change_tok1_mean"] = float(tot[:, 1].mean()); out["score_change_tok2_mean"] = float(tot[:, 2].mean())
gapc = pp[:, 2] - pp[:, 1]                    # change of (score t2 - score t1): the redirect direction
G = gapc.mean(0).numpy()
out["per_pair_gapchange_mean"] = G.tolist()
out["band_share_gapchange"] = float(G[61:64].sum() / G.sum())
order = np.argsort(-np.abs(G))
out["top8_pairs_abs_gapchange"] = [[int(j), float(G[j])] for j in order[:8]]
dkn = (dk[:, 1:] ** 2).sum((0, 1)); e = (dkn[:64] + dkn[64:]).numpy()
out["band_share_dk_energy_t12"] = float(e[61:64].sum() / e.sum())
out["attn_on_ans_t0_t1_t2"] = rec_on[1][:, 8, -1, :3].mean(0).tolist()
out["attn_off_ans_t0_t1_t2"] = rec_off[1][:, 8, -1, :3].mean(0).tolist()
# exact head-27.8 attention when only the band's score terms change (other heads, values, other layers fixed)
w_on = (q[:, None, :] * cap_on[1][1][:, 8].float()).sum(-1) * sc   # [N,T] cue-on scores at the answer row
for name, band in (("band61_63", [61, 62, 63]), ("all64", list(range(64)))):
    add = torch.zeros_like(w_on); add[:, :3] = pp[..., band].sum(-1)
    a = torch.softmax(w_on + add, -1)[:, :3].mean(0)
    out[f"head27_8_attn_t012_{name}"] = a.tolist()
out["sec"] = time.time() - t0
json.dump(out, open(f"{V.HERE}/vera_gate_{seed}_{dt}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_pair_gapchange_mean"}, indent=1))

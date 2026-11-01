"""Vera's sink-vs-token-2 arms + R6 split (own code; frozen in room #11251 before running).
Usage: python vera_kv3.py FAMILY SEED DTYPE DEV OUT.npz
A2SINK  : recipient attention; token 2's weight set to its donor-K value; the difference moved onto token 0.
A2VANISH: same token-2 change; the difference goes nowhere (weights no longer sum to 1; a zero-value sink).
Both for all 32 heads, and for the top K head alone. Rows >= 3, both blocks.
R6 (block 27, answer token, first order): per query head, contributions g . W_O^h (delta a_j v_on_j) [K-term, j = 0,
1, 2, rest] and g . W_O^h (a_on_j (v_off_j - v_on_j)) [V-term, j = 0, 1-2], with delta a from donor K at tokens 1-2
(K-term) or 0-2 (K02), g = d log-odds P(X) / d (block-27 attention output at the answer token), recipient run.
Also: base (untrained) answer-token attention per head on tokens 0/1/2, cue on and off.
"""
import sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_eng import Eng, S, NH, BS
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
fam, seed, dt, dev, out = sys.argv[1:6]
t0 = time.time(); res = {}
base = Eng(fam, "base", "fp32", "cpu")
for nm, h in [("on", base.CON), ("off", base.COFF)]:
    a = base.attn(h)
    for L in (0, 1): res[f"base_att_{nm}_ans_L{L}"] = a[f"ans_L{L}"][:, :3]
del base; import gc; gc.collect()
E = Eng(fam, seed, dt, dev); M = E.M; ALL = list(range(NH))
res["on"], res["off"] = E.plain(E.CON), E.plain(E.COFF)
res["k@1-2"] = E.patch(lambda: {L: [("k", M(ALL, [1, 2]))] for L in (0, 1)})
res["k@2"] = E.patch(lambda: {L: [("k", M(ALL, [2]))] for L in (0, 1)})
res["k@1"] = E.patch(lambda: {L: [("k", M(ALL, [1]))] for L in (0, 1)})
res["A2SINK"] = E.patch(lambda: {L: [("a2sink", M(ALL, [2]))] for L in (0, 1)})
res["A2VANISH"] = E.patch(lambda: {L: [("a2vanish", M(ALL, [2]))] for L in (0, 1)})
# top K head: largest single-head K-only share at 0-2 from my vera2 file of this copy
z2 = np.load(f"vera2_{fam}_{seed}_{dt}_mps.npz")
sh = lambda d, z: (d.mean() - z["on"].mean()) / (z["off"].mean() - z["on"].mean())
hk = np.array([sh(z2[f"k@0-2_L{L}h{h}"], z2) for L in (0, 1) for h in range(NH)]); top = int(np.argmax(hk)); TL, TH = top // NH, top % NH
res["top_head"] = np.array([TL, TH]); print("top K head", 26 + TL, TH, round(float(hk[top]), 3), flush=True)
res["A2SINK_top"] = E.patch(lambda: {TL: [("a2sink", M([TH], [2]))]})
res["A2VANISH_top"] = E.patch(lambda: {TL: [("a2vanish", M([TH], [2]))]})
res["k@1-2_top"] = E.patch(lambda: {TL: [("k", M([TH], [1, 2]))]})
print("arms", round(time.time() - t0), "s", flush=True)
# R6 first-order split at block 27 (layer index 1)
att = E.model.model.layers[1].self_attn; WO = att.o_proj.weight.float()  # 1024 x 2048
K = {"Kterm12_j0": [], "Kterm12_j1": [], "Kterm12_j2": [], "Kterm12_rest": [], "Kterm02_j0": [], "Kterm02_j12": [], "Kterm02_rest": [],
     "Vterm_j0": [], "Vterm_j12": []}
for i in range(0, len(E.CON), BS):
    S["mode"] = "capture"; S["donor"] = {}; S["q"] = {}; E.run(E.COFF[i:i + BS]); dk, dv = [t.float() for t in S["donor"][1]]
    S["mode"] = "capture"; S["donor"] = {}; S["q"] = {}; E.run(E.CON[i:i + BS]); rk, rv = [t.float() for t in S["donor"][1]]; q = S["q"][1].float()
    S["mode"] = "none"
    box = {}
    def hk_(m, a, o): o[0].retain_grad(); box["o"] = o[0]
    hnd = att.register_forward_hook(hk_)
    lg = E.model(inputs_embeds=E.CON[i:i + BS].to(E.DT).to(E.dev), use_cache=False, logits_to_keep=1).logits[:, -1].float()
    lp = torch.log_softmax(lg, -1); lpx = torch.logsumexp(lp[:, E.X.to(lg.device)], -1)
    lo = lpx - torch.log1p(-lpx.exp()); lo.sum().backward(); hnd.remove()
    g = box["o"].grad[:, -1].float()  # N x 1024
    E.model.zero_grad(set_to_none=True)
    rk, rv, dk, dv = [repeat_kv(t, 2) for t in (rk, rv, dk, dv)]  # N,16,T,128
    qa = q[:, :, -1]  # N,16,128
    def a_of(k): return torch.softmax(torch.einsum("nhd,nhtd->nht", qa, k) * att.scaling, -1)
    a_on = a_of(rk)
    k12 = rk.clone(); k12[:, :, 1:3] = dk[:, :, 1:3]; k02 = rk.clone(); k02[:, :, :3] = dk[:, :, :3]
    WOh = WO.view(1024, NH, 128)
    def proj(vecs):  # vecs N,16,128 -> N,16 contribution g . W_O^h vec
        return torch.einsum("nc,chd,nhd->nh", g, WOh, vecs)
    for tag, kk in [("12", k12), ("02", k02)]:
        da = a_of(kk) - a_on  # N,16,T
        c = {j: proj(da[:, :, j, None] * rv[:, :, j]) for j in (0, 1, 2)}
        rest = proj((da[:, :, 3:, None] * rv[:, :, 3:]).sum(2))
        if tag == "12":
            K["Kterm12_j0"].append(c[0]); K["Kterm12_j1"].append(c[1]); K["Kterm12_j2"].append(c[2]); K["Kterm12_rest"].append(rest)
        else:
            K["Kterm02_j0"].append(c[0]); K["Kterm02_j12"].append(c[1] + c[2]); K["Kterm02_rest"].append(rest)
    K["Vterm_j0"].append(proj(a_on[:, :, 0, None] * (dv[:, :, 0] - rv[:, :, 0])))
    K["Vterm_j12"].append(proj((a_on[:, :, 1:3, None] * (dv[:, :, 1:3] - rv[:, :, 1:3])).sum(2)))
for k, v in K.items(): res[f"R6_{k}"] = torch.cat(v).detach().cpu().numpy()  # N x 16 (block 27 heads)
np.savez(out, **res)
print("saved", out, round(time.time() - t0), "s", time.strftime("%H:%M:%S"), flush=True)

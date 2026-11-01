"""Vera's status + null rows with my own engine (vera_eng.py). Own code; build's scripts not read.
Usage: python vera_kv2.py FAMILY SEED DTYPE DEV OUT.npz
Rows: on, off, noop, where-noop (recipient K/V swapped in through the patch path, the layout-noise floor), resid@0-2
(gate), kv/k/v x {0-2, 0, 1-2}; per-head v and k at 0-2 (32 heads); SV (my sink set: base tail, condition-on input,
fp32 CPU, mean attention on tokens 0-2 over query tokens 3-33 > 0.5), NV/NK (the other heads), 99 random size-matched
V sets (rng 20261101, per block, ties with the sink set redrawn), R5 touchiness (same-norm random V delta on SV heads,
99 draws), R8 sink-held K at tokens 1-2 (all heads), answer-token attention per head on and off (trained copy).
"""
import sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_eng import Eng, S, NH
fam, seed, dt, dev, out = sys.argv[1:6]
t0 = time.time()
base = Eng(fam, "base", "fp32", "cpu")
ab = base.attn(base.CON)
mass = np.stack([ab["all_L0"][:, :3].sum(1), ab["all_L1"][:, :3].sum(1)])  # 2 x 16
SINK = [[h for h in range(NH) if mass[L, h] > 0.5] for L in (0, 1)]
print("sink set", [len(s) for s in SINK], "non-sink", [[h for h in range(NH) if h not in SINK[L]] for L in (0, 1)], flush=True)
del base; import gc; gc.collect()
E = Eng(fam, seed, dt, dev); M = E.M
res = {"base_mass": mass}
for nm, h in [("on", E.CON), ("off", E.COFF)]:
    a = E.attn(h)
    for k, v in a.items(): res[f"att_{nm}_{k}"] = v
res["on"], res["off"] = E.plain(E.CON), E.plain(E.COFF)
res["noop"] = E.patch(lambda: {})
ALL = list(range(NH)); P02, P0, P12 = [0, 1, 2], [0], [1, 2]
res["wherenoop"] = E.patch(lambda: {L: [("kv", M(ALL, P02))] for L in (0, 1)}, don=E.CON)
g = E.CON.clone(); g[:, :3] = E.COFF[:, :3]; res["resid@0-2"] = E.plain(g)
for pn, pos in [("0-2", P02), ("0", P0), ("1-2", P12)]:
    for w in ["kv", "k", "v"]:
        res[f"{w}@{pn}"] = E.patch(lambda pos=pos, w=w: {L: [(w, M(ALL, pos))] for L in (0, 1)})
for L in (0, 1):
    for hd in range(NH):
        for w in ["v", "k"]:
            res[f"{w}@0-2_L{L}h{hd}"] = E.patch(lambda L=L, hd=hd, w=w: {L: [(w, M([hd], P02))]})
print("main rows", round(time.time() - t0), "s", flush=True)
res["SV"] = E.patch(lambda: {L: [("v", M(SINK[L], P02))] for L in (0, 1)})
NS = [[h for h in ALL if h not in SINK[L]] for L in (0, 1)]
res["NV"] = E.patch(lambda: {L: [("v", M(NS[L], P02))] for L in (0, 1) if NS[L]})
res["NK"] = E.patch(lambda: {L: [("k", M(NS[L], P02))] for L in (0, 1) if NS[L]})
res["SK"] = E.patch(lambda: {L: [("k", M(SINK[L], P02))] for L in (0, 1)})
rng = np.random.default_rng(20261101); sets = []
while len(sets) < 99:
    s = [sorted(rng.choice(NH, len(SINK[L]), replace=False).tolist()) for L in (0, 1)]
    if s != SINK: sets.append(s)
res["rand_sets"] = np.array([[int(h in s[L]) for L in (0, 1) for h in range(NH)] for s in sets])
for r, s in enumerate(sets):
    res[f"RV{r:02d}"] = E.patch(lambda s=s: {L: [("v", M(s[L], P02))] for L in (0, 1)})
print("random sets", round(time.time() - t0), "s", flush=True)
# R5: same-norm random direction added to the recipient V at tokens 0-2 on the SV heads (per head, token, prompt)
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
g5 = torch.Generator().manual_seed(4)
for r in range(99):
    def vr(i, r=r):
        out = {}
        S["mode"] = "capture"; S["donor"] = {}; E.run(E.CON[i:i + 50]); rec = {L: S["donor"][L][1] for L in (0, 1)}
        S["mode"] = "capture"; S["donor"] = {}; E.run(E.COFF[i:i + 50]); don = {L: S["donor"][L][1] for L in (0, 1)}
        S["mode"] = "patch"
        for L in (0, 1):
            d = repeat_kv(don[L] - rec[L], 2).float()  # N, 16, T, 128
            z = torch.randn(d.shape, generator=g5).to(d.device)
            z = z / z.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
            out[L] = z.to(E.DT)
        return out
    res[f"R5_{r:02d}"] = E.patch(lambda: {L: [("vrand", M(SINK[L], P02))] for L in (0, 1)}, vrand_fn=vr)
res["R8_khold@1-2"] = E.patch(lambda: {L: [("khold", M(ALL, P12))] for L in (0, 1)})
res["R8_k@1-2"] = res["k@1-2"]
np.savez(out, **res)
print("saved", out, round(time.time() - t0), "s", time.strftime("%H:%M:%S"), flush=True)

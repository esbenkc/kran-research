"""Phase 0 (PREREG v1, before any pinned run): refusal direction r, every unit's write vector, and the dose
numbers p, d, p_last, u for every (unit, pin) from CLEAN passes only.
Run: env/.venv/bin/python -u swarm-theory-1031-refusal-dose-law/phase0.py
Writes data/phase0/{units.pt, p_table.json, phase0.json, r13.npy, prompt_stats.pt}."""
import json
import time

import numpy as np
import torch

from common import DEV, GATE, L, P, chat, down, encode, load_model

OUT = P / "data/phase0"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((P / "data/splits.json").read_text())
M5 = [6.0, 12.0, 18.0, 24.0, 30.0]
tok, model = load_model()
W = down(model).weight.float()  # [2048, 6144]
d_model = W.shape[0]
t0 = time.time()


@torch.no_grad()
def last_states(prompts, layers, bs=50):
    """Residual at the output of each layer in `layers`, last prompt token, fp32 [n, len(layers), d]."""
    out = []
    for s in range(0, len(prompts), bs):
        enc = encode(tok, [chat(tok, p) for p in prompts[s:s + bs]])
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([hs[l + 1][:, -1].float() for l in layers], 1).cpu())
    return torch.cat(out)


# ---- r (Vera's recipe): L13 output, last prompt token, dev_harmful[:100] minus test_harmless[:100]; all layers for L*
ALL = list(range(model.config.num_hidden_layers))
Hh = last_states(S["dev_harmful"][:100], ALL)
Hn = last_states(S["test_harmless"][:100], ALL)
R = Hh.mean(0) - Hn.mean(0)  # [layers, d]
r = R[L]
r_hat = r / r.norm()
np.save(OUT / "r13.npy", r.numpy())
info = {"r_norm": r.norm().item(), "mean_norm_harm_last": Hh[:, L].norm(dim=-1).mean().item()}
try:
    rv = torch.tensor(np.load(P / "skeptic/vera_r13.npy")).float().flatten()
    info["cos_r_bruno_r_vera"] = torch.nn.functional.cosine_similarity(r, rv, 0).item()
except Exception as e:  # noqa
    info["cos_r_bruno_r_vera"] = f"n/a: {e}"
# L*: argmax over layers 14..27 of DEV separation (mean projection gap / pooled sd) along r_hat_l
sep = {}
for l in range(L + 1, len(ALL)):
    rh = R[l] / R[l].norm()
    a, b = Hh[:, l] @ rh, Hn[:, l] @ rh
    sep[l] = ((a.mean() - b.mean()) / torch.sqrt((a.var() + b.var()) / 2)).item()
LSTAR = max(sep, key=sep.get)
info.update(sep_by_layer=sep, L_star=LSTAR)
torch.save({"r_hat_Lstar": (R[LSTAR] / R[LSTAR].norm()), "L_star": LSTAR}, OUT / "lstar.pt")

# ---- cos of every layer-13 neuron write with r_hat
cos = (W.T.cpu() @ r_hat) / W.norm(dim=0).cpu()
order = torch.argsort(cos)
info["gate_cos"] = cos[GATE].item()
info["gate_rank_abs"] = int((cos.abs() > abs(cos[GATE])).sum()) + 1
info["cos_abs_median"] = cos.abs().median().item()
info["cos_abs_p99"] = cos.abs().quantile(0.99).item()
neg = [int(i) for i in order[:6] if int(i) != GATE][:4]
pos = [int(i) for i in order.flip(0)[:2]]
info["NN_neg"], info["NN_pos"] = neg, pos
info["NN_cos"] = {i: cos[i].item() for i in neg + pos}
info["NN_vera"] = [1265, 3990, 4786, 2142, 5861, 2591]
print(info["cos_r_bruno_r_vera"], info["gate_cos"], info["gate_rank_abs"], neg, pos, "L*", LSTAR, flush=True)

# ---- unit write vectors (per unit of coefficient). Gate-coefficient units: delta_t = (m - a_t^gate) * vec.
w = W[:, GATE].cpu()
wn = w.norm()
c_g = cos[GATE].item()
s = float(np.sign(w @ r_hat))


def orth_unit(seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d_model, generator=g)
    u = u - (u @ r_hat) * r_hat
    return u / u.norm()


def dialed(c, seed):
    return wn * (s * c * r_hat + (1 - c * c) ** 0.5 * orth_unit(seed))


units = {"G": ("gate", w), "G_on": ("gate", (w @ r_hat) * r_hat), "G_off": ("gate", w - (w @ r_hat) * r_hat)}
for k in range(3):  # TWIN: same on-axis part as the gate, random orthogonal part of the same length (u seeds 100-102)
    units[f"TWIN_{k}"] = ("gate", dialed(abs(c_g), 100 + k))
for c in [0.05, 0.1, 0.2, 0.6, 1.0]:  # one u seed (200) for every c
    units[f"D_{c}"] = ("gate", dialed(c, 200))
rng = np.random.default_rng(0)
RN = [int(j) for j in rng.choice([j for j in range(W.shape[1]) if j != GATE], 100, replace=False)]
for j in RN:
    units[f"RN_{j}"] = ("gate", wn * W[:, j].cpu() / W[:, j].norm().cpu())
rd = np.random.default_rng(1).standard_normal((20, d_model))
for k in range(20):
    u = torch.tensor(rd[k], dtype=torch.float32)
    units[f"RD_{k}"] = ("gate", wn * u / u.norm())
# FK: DoM of two random halves (seed 2) of test_harmless[100:200], same layer and token
Hf = last_states(S["test_harmless"][100:200], [L])[:, 0]
perm = np.random.default_rng(2).permutation(100)
rf = Hf[perm[:50]].mean(0) - Hf[perm[50:]].mean(0)
units["FK"] = ("gate", wn * rf / rf.norm())
info["FK_cos_r"] = (rf @ r_hat / rf.norm()).item()
for j in neg + pos:  # natural neurons, native pin
    units[f"NN_{j}"] = ("native", j)
info["RN"] = RN

# ---- clean pass on TEST: per prompt, tokens t >= 1 of the prompt: ||h_t||, gate activation, native activations
NAT = sorted(set(neg + pos + RN[:3]))
stats = []
cap = {}
hk = down(model).register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0]))
with torch.no_grad():
    texts = [chat(tok, p) for p in S["test_harmful"]]
    for i in range(0, len(texts), 25):
        enc = encode(tok, texts[i:i + 25])
        hs = model(**enc, output_hidden_states=True).hidden_states
        h = hs[L + 1].float()
        x = cap["x"].float()
        for b in range(h.shape[0]):
            mk = enc["attention_mask"][b].bool()
            pos_ = torch.nonzero(mk).flatten()[1:]  # skip token 0
            hb = h[b, pos_].cpu()
            stats.append({"hn": hb.norm(dim=-1), "a_g": x[b, pos_, GATE].cpu(),
                          "a_nat": x[b][pos_][:, NAT].cpu(), "h_last_norm": hb[-1].norm()})
hk.remove()
info["NAT"] = NAT
torch.save({"stats": stats, "NAT": NAT}, OUT / "prompt_stats.pt")


def dose(kind, vec_or_j, m, nat_scale=None):
    """Per-prompt arrays (n=200): p, d, p_last, u for the edit."""
    ps, ds, pl, us = [], [], [], []
    for st in stats:
        if kind == "gate":
            coef, vec = (m - st["a_g"]), vec_or_j
        else:
            j = vec_or_j
            coef, vec = (m - st["a_nat"][:, NAT.index(j)]), W[:, j].cpu()
        proj = -(coef * (vec @ r_hat))  # [T]
        ps.append((proj / st["hn"]).mean().item())
        ds.append((coef.abs() * vec.norm() / st["hn"]).mean().item())
        pl.append((proj[-1] / st["h_last_norm"]).item())
        us.append((proj[-1] / r.norm()).item())
    return {"p": ps, "d": ds, "p_last": pl, "u": us}


table = {}
for name, (kind, v) in units.items():
    for m in M5:
        table[f"{name}@{m:g}"] = dose(kind, v, m)
# PM: 3 random neurons (RN[:3]), native pin m_j so that p over the first 50 TEST prompts equals the gate's at m=18, 30
pm = {}
for j in RN[:3]:
    for mg in [18.0, 30.0]:
        target = np.mean(table[f"G@{mg:g}"]["p"][:50])
        p0 = np.mean(dose("native", j, 0.0)["p"][:50])
        p1 = np.mean(dose("native", j, 1.0)["p"][:50])
        mj = (target - p0) / (p1 - p0)  # p is linear in m
        units[f"PM_{j}_{mg:g}"] = ("native", j)
        table[f"PM_{j}_{mg:g}@{mj:.4f}"] = dose("native", j, mj)
        pm[f"PM_{j}_{mg:g}"] = mj
info["PM_pins"] = pm
torch.save({k: (v[0], v[1] if isinstance(v[1], int) else v[1].clone()) for k, v in units.items()}, OUT / "units.pt")
(OUT / "p_table.json").write_text(json.dumps(table))
summ = {k: {q: float(np.mean(v[q])) for q in v} for k, v in table.items() if k.split("@")[0] in
        ["G", "G_on", "G_off", "TWIN_0", "D_0.05", "D_0.2", "D_1.0", f"RN_{RN[0]}", "RD_0", "FK"]}
info["p_summary_mean_test200"] = summ
info["seconds"] = time.time() - t0
(OUT / "phase0.json").write_text(json.dumps(info, indent=1))
for k, v in summ.items():
    print(k, {q: round(x, 4) for q, x in v.items()}, flush=True)
print("done", round(time.time() - t0), "s")

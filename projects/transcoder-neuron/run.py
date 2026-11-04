"""READ and WRITE arms of PREREG v2.1 for one layer.

  env/.venv/bin/python run.py --layer 8            (full: 100 units/seed/group, 3 seeds, 8 contexts)
  env/.venv/bin/python run.py --layer 8 --smoke    (10 units/seed/group, 1 seed, 4 contexts)

Writes data/pushes/L{L}{_smoke}_units.csv (one row per unit) and data/pushes/L{L}{_smoke}_pushes.csv (one row per
(unit, context)), plus data/pushes/L{L}{_smoke}_meta.json. Column meanings: see COLUMNS below.

Units (groups): NEURON (enc AND dec |cos| >= 0.9 to the same neuron), REST (max enc |cos| < 0.5 and max dec
|cos| < 0.5), GAP (0.5 <= joint < 0.9, 25 per 0.1 bin per seed), ORACLE (a real GPT-2 neuron used as a feature:
e = g2 * W_fc[:, n], d = centred W_proj[n], pre-activation z_n, ReLU-style gate z_n > 0).
Each unit: seed s in {0,1,2}; features drawn uniformly (seeded) from those active on the windows.
Contexts: up to C active (window, position) pairs per unit drawn uniformly at random; t in 1..126.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import DEV, GPT2, centred, gelu_new, gelu_new_grad

COLUMNS = """
units: L group seed uid (feature or neuron index) enc_cos dec_cos joint same_neuron max_enc_any max_dec_any
       e_norm d_norm n_active fire_rate mean_act
pushes (y = centred logit of the true next token at position t; g = d y / d mlp_out_L at the clean input:
        g_graph through the circuit-tracer linearisation, g_real through the real model):
  pre        unit pre-activation at the context (feature: a_f; oracle: z_n)
  du_norm, xn_norm
  meas_local_g   g_graph . (MLP(x+du) - MLP(x)),      du = -pre e / |e|^2 (turn the unit off)
  total_g        g_graph . (TC(x+du) - TC(x))          (PREREG v2.6: PRIMARY is the *_r pair: total_r vs meas_local_r)
  jvp_g          g_graph . (J_TC du)  (active set frozen)
  fonly_g        f-only node claim: -a_f d_f . g_graph  (oracle: (gelu(0)-gelu(z)) d_n . g_graph)
  floor_g        -a_f d_rand . g_graph, |d_rand| = |d_f| (FLOOR)
  *_r            the same four with g_real (isolation row, Kasper #14382)
  mir_meas_g, mir_total_g     mirror dose +pre e/|e|^2 (and *_r with g_real)
  rnd_meas_g, rnd_total_g     random push direction at |du| (NULL) (and *_r with g_real)
  cos_vec        cos(MLP(x+du)-MLP(x), TC(x+du)-TC(x)) in d_model (centred MLP output)
  tr2, r2        |t - r|^2 and |r|^2, t = TC(x+du)-TC(x), r = MLP(x+du)-MLP(x) (Prediction C); *_rnd for du_rnd
  flips          number of TC features whose on/off state changes under du (includes f itself); flips_rnd
  curv           |r - J_MLP du| / |J_MLP du| (curvature of the real MLP along du); curv_rnd
  rank1          d_f^T J_MLP e_f / (|d_f| |e_f| |J_MLP|_F) at x (2610.24521's alignment)
  e2e_real       y(real model, MLP_L input x+du) - y(clean)
  e2e_tc         y(MLP_L out := MLP(x) + TC(x+du) - TC(x)) - y(clean)
  e2e_rnd        y(real model, MLP_L input x+du_rnd) - y(clean)
  wr_minus, wr_plus      WRITE: y(MLP_L out := TC(x) -/+ a d) - y(MLP_L out := TC(x))
  wr_rnd_minus, wr_rnd_plus   same with a random direction at |a d|
  wr_pred        a d . g_graph (WRITE pred for +; - for minus)
  wr_rnd_pred    a d_rnd . g_graph;  wr_pred_r, wr_rnd_pred_r with g_real
  g_cos, g_norm_ratio   cos(g_graph, g_real), |g_graph| / |g_real|
"""

def fp():
    import re, subprocess
    o = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    mm = re.search(r"Footprint:\s*([\d.]+\s*(B|KB|MB|GB))", o)
    return mm.group(1) if mm else "?"


ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--per-seed", type=int, default=100)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--ctx", type=int, default=8)
ap.add_argument("--windows", type=int, default=2000)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--tok", type=int, default=4096)
args = ap.parse_args()
if args.smoke:
    args.per_seed, args.seeds, args.ctx = 10, 1, 4
L = args.layer
tag = f"L{L}" + ("_smoke" if args.smoke else "")
os.makedirs("data/pushes", exist_ok=True)
t0 = time.time()
torch.manual_seed(1000 + L)
rng = np.random.default_rng(1000 + L)

# ---------------------------------------------------------------- windows
WPATH = "data/windows.npy"
if not os.path.exists(WPATH):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    text = load_dataset("NeelNanda/pile-10k", split="train")["text"]
    rows, docs = [], []
    for i in range(5000, 10000):
        t = tok(text[i][:2000])["input_ids"]
        if len(t) >= 127:
            rows.append([50256] + t[:127]); docs.append(i)
        if len(rows) == 2000:
            break
    np.save(WPATH, np.array(rows, dtype=np.int32))
    json.dump({"doc_index": docs, "rule": "pile-10k docs 5000.. in index order, first 2000 with >=127 tokens, "
               "BOS + 127 tokens (first 2000 chars tokenised)"}, open("data/windows_meta.json", "w"))
W = torch.tensor(np.load(WPATH)[: args.windows], dtype=torch.long)
NW, T = W.shape

print(f"[{tag}] start fp {fp()}", flush=True)
m = GPT2()
tc = m.tc(L)
for l in range(L + 1, 12):
    m.tc(l)
print(f"[{tag}] weights fp {fp()}", flush=True)
b = m.blocks[L]
g2 = b["ln_2.weight"]
Wfc, bfc = b["mlp.c_fc.weight"], b["mlp.c_fc.bias"]
Wpc = b["mlp.c_proj.weight"] - b["mlp.c_proj.weight"].mean(1, keepdim=True)  # centred W_out rows


Qn = (g2[:, None] * Wfc).T  # [3072, 768]
KJ = (Wpc @ Wpc.T) * (Qn @ Qn.T)  # [3072, 3072]


def mlp_c(xn):
    """centred MLP_L output (mean over d_model removed; invisible downstream)."""
    o = m.mlp(b, xn)
    return o - o.mean(-1, keepdim=True)


def zpre(xn):
    return (xn * g2 + b["ln_2.bias"]) @ Wfc + bfc


# ---------------------------------------------------------------- cosines and classes
# PREREG v2.1: enc_cos = max_n |cos(e_f, W_in n)|, n* = argmax; dec_cos = |cos(d_f, W_out_c[n*])|.
with torch.no_grad():
    Win = g2[:, None] * Wfc
    Win = Win - Win.mean(0, keepdim=True)
    Win = Win / Win.norm(dim=0, keepdim=True)
    Wo = Wpc / Wpc.norm(dim=1, keepdim=True)
    enc_cos, nstar, dec_cos, max_enc, max_dec = [], [], [], [], []
    for s in range(0, tc.W_enc.shape[1], 2048):  # chunked: no full normalised copies of the dictionary
        E = tc.W_enc[:, s:s + 2048]; E = E - E.mean(0, keepdim=True); E = E / E.norm(dim=0, keepdim=True)
        Dd = tc.W_dec[s:s + 2048]; Dd = Dd - Dd.mean(1, keepdim=True); Dd = Dd / Dd.norm(dim=1, keepdim=True)
        ce = (E.T @ Win).abs()
        cd = (Dd @ Wo.T).abs()
        v, i = ce.max(1)
        enc_cos.append(v); nstar.append(i); max_enc.append(v)
        dec_cos.append(cd.gather(1, i[:, None]).squeeze(1)); max_dec.append(cd.max(1).values)
    enc_cos, nstar, dec_cos = torch.cat(enc_cos).cpu(), torch.cat(nstar).cpu(), torch.cat(dec_cos).cpu()
    max_enc, max_dec = torch.cat(max_enc).cpu(), torch.cat(max_dec).cpu()
del E, Dd, ce, cd
torch.mps.empty_cache()
print(f"[{tag}] cosines fp {fp()}", flush=True)
joint = torch.minimum(enc_cos, dec_cos)
NEU = (enc_cos >= 0.9) & (dec_cos >= 0.9)
RST = (max_enc < 0.5) & (max_dec < 0.5)

# ---------------------------------------------------------------- pass 1: firing stats
F = tc.W_enc.shape[1]
cnt = torch.zeros(F, dtype=torch.float64); sact = torch.zeros(F, dtype=torch.float64)
ncnt = torch.zeros(3072, dtype=torch.float64)
ntok = 0
with torch.no_grad():
    for s in range(0, NW, 8):
        ids = W[s:s + 8].to(DEV)
        _, cap = m.forward(ids, capture={f"xn{L}"}, stop_after=L)
        xn = cap[f"xn{L}"][:, 1:T - 1]  # positions 1..126
        a = tc.acts(xn).reshape(-1, F)
        cnt += (a > 0).sum(0).cpu().double(); sact += a.sum(0).cpu().double()
        ncnt += (zpre(xn).reshape(-1, 3072) > 0).sum(0).cpu().double()
        ntok += a.shape[0]
alive = cnt > 0
torch.mps.empty_cache()
print(f"[{tag}] pass1 fp {fp()} {time.time()-t0:.0f}s ntok={ntok} NEURON {int(NEU.sum())} (active {int((NEU&alive).sum())}) "
      f"REST {int(RST.sum())} (active {int((RST&alive).sum())})", flush=True)

# ---------------------------------------------------------------- selection
groups = {"NEURON": NEU & alive, "REST": RST & alive}
units = []  # (group, seed, uid)
need = args.per_seed * args.seeds
for gname, mask in groups.items():
    cand = np.flatnonzero(mask.numpy())
    cand = rng.permutation(cand)[: need]
    per = len(cand) // args.seeds
    for s in range(args.seeds):
        for u in cand[s * per:(s + 1) * per]:
            units.append((gname, s, int(u)))
# GAP zone (PREREG v2.3 Sampling): per seed 25 active features per 0.1 bin of joint cosine in [0.5, 0.9);
# a short bin is filled from the other bins (uniformly from the leftover pool of the zone).
jn = joint.numpy(); al = alive.numpy()
gap_bins = [rng.permutation(np.flatnonzero(al & (jn >= lo) & (jn < lo + 0.1) & ~NEU.numpy())) for lo in (0.5, 0.6, 0.7, 0.8)]
per_bin = args.per_seed // 4
gap_short = {}
for s in range(args.seeds):
    picked = []
    for bi, pool in enumerate(gap_bins):
        take = pool[s * per_bin:(s + 1) * per_bin]
        picked += list(take)
        if len(take) < per_bin:
            gap_short[f"seed{s}_bin{bi}"] = int(len(take))
    if len(picked) < args.per_seed:  # fill from the zone's unused features
        used = set(int(u) for g_, s_, u in units if g_ == "GAP") | set(int(u) for u in picked)
        rest_pool = [int(u) for pool in gap_bins for u in pool[args.seeds * per_bin:] if int(u) not in used]
        picked += rest_pool[: args.per_seed - len(picked)]
    for u in picked:
        units.append(("GAP", s, int(u)))
ncand = rng.permutation(np.flatnonzero((ncnt > 0).numpy()))[: need]
per = len(ncand) // args.seeds
for s in range(args.seeds):
    for u in ncand[s * per:(s + 1) * per]:
        units.append(("ORACLE", s, int(u)))
feat_units = [u for u in units if u[0] != "ORACLE"]
fidx = torch.tensor([u[2] for u in feat_units], dtype=torch.long)
nidx = torch.tensor([u[2] for u in units if u[0] == "ORACLE"], dtype=torch.long)

# ---------------------------------------------------------------- pass 2: contexts
# Draw C occurrence ranks per unit from its pass-1 count, then pick the entries with those ranks
# (uniform over the unit's active positions, no list of all occurrences kept).
U = len(units)
counts = np.array([int(ncnt[u]) if g == "ORACLE" else int(cnt[u]) for g, s_, u in units])
chosen = np.full((U, args.ctx), -1, dtype=np.int64)
for k, n in enumerate(counts):
    p = rng.choice(n, size=min(args.ctx, n), replace=False)
    chosen[k, : len(p)] = p
chosen_t = torch.tensor(chosen, device=DEV)
running = torch.zeros(U, dtype=torch.long, device=DEV)
contexts = []
nf = len(feat_units)
with torch.no_grad():
    We_s, be_s = tc.W_enc[:, fidx.to(DEV)], tc.b_enc[fidx.to(DEV)]
    for s in range(0, NW, 8):
        ids = W[s:s + 8].to(DEV)
        _, cap = m.forward(ids, capture={f"xn{L}"}, stop_after=L)
        xn = cap[f"xn{L}"][:, 1:T - 1]
        a = (xn - tc.b_dec) @ We_s + be_s
        z = zpre(xn)[..., nidx.to(DEV)]
        mask = (torch.cat([a, z], -1) > 0).reshape(-1, U)  # [N, U]
        rank = running[None] + mask.long().cumsum(0) - 1
        i, k = mask.nonzero(as_tuple=True)
        r = rank[i, k]
        hit = (chosen_t[k] == r[:, None]).any(-1)
        for ii, kk in zip(i[hit].tolist(), k[hit].tolist()):
            contexts.append((kk, s + ii // (T - 2), ii % (T - 2) + 1))
        running += mask.sum(0)
torch.mps.empty_cache()
print(f"[{tag}] pass2 fp {fp()} {time.time()-t0:.0f}s units={len(units)} pushes={len(contexts)} "
      f"count-mismatch units={int((running.cpu().numpy() != counts).sum())}", flush=True)

# ---------------------------------------------------------------- unit vectors
E_units, D_units = [], []
for gname, s, u in units:
    if gname == "ORACLE":
        E_units.append(g2 * Wfc[:, u]); D_units.append(Wpc[u])
    else:
        E_units.append(tc.W_enc[:, u]); D_units.append(tc.W_dec[u] - tc.W_dec[u].mean())
E_units = torch.stack(E_units); D_units = torch.stack(D_units)  # [U, 768]

# ---------------------------------------------------------------- pass 3: pushes
ctx = np.array(contexts)  # k, w, t
order = np.argsort(ctx[:, 2], kind="stable")
ctx = ctx[order]
out = {c: np.zeros(len(ctx)) for c in [
    "pre", "du_norm", "xn_norm", "meas_local_g", "total_g", "jvp_g", "fonly_g", "floor_g", "meas_local_r",
    "total_r", "jvp_r", "fonly_r", "floor_r", "mir_meas_g", "mir_total_g", "rnd_meas_g", "rnd_total_g",
    "mir_meas_r", "mir_total_r", "rnd_meas_r", "rnd_total_r", "g_cos", "g_norm_ratio", "wr_pred_r", "wr_rnd_pred_r", "cos_vec",
    "rank1", "e2e_real", "e2e_tc", "e2e_rnd", "wr_minus", "wr_plus", "wr_rnd_minus", "wr_rnd_plus", "wr_pred",
    "wr_rnd_pred", "y_clean", "y_full_minus_cache", "pre_ok", "tr2", "r2", "tr2_rnd", "r2_rnd",
    "flips", "flips_rnd", "curv", "curv_rnd"]}
gen = torch.Generator().manual_seed(2000 + L)
isor = torch.tensor([units[k][0] == "ORACLE" for k in ctx[:, 0]])
# batches with a token budget (rows x prefix length <= --tok), since the footprint grows with the prefix length
starts = []
s = 0
while s < len(ctx):
    e_ = s + 1
    while e_ < len(ctx) and (e_ - s + 1) * (int(ctx[e_, 2]) + 1) <= args.tok and e_ - s < args.bs:
        e_ += 1
    starts.append((s, e_)); s = e_
for bi, (s, e_) in enumerate(starts):
    sl = slice(s, e_)
    c = ctx[sl]
    B = len(c)
    Tm = int(c[:, 2].max()) + 1
    ids = W[c[:, 1], :Tm].to(DEV)
    tpos = torch.tensor(c[:, 2], device=DEV)
    rows = torch.arange(B, device=DEV)
    nxt = W[c[:, 1], c[:, 2] + 1].to(DEV)
    e = E_units[torch.tensor(c[:, 0], device=DEV)]; d = D_units[torch.tensor(c[:, 0], device=DEV)]
    orc = isor[sl].to(DEV)
    with torch.no_grad():
        lg_full, _ = m.forward(ids, read_pos=(rows, tpos))
        _, cap = m.forward(ids, capture={f"xn{L}"}, stop_after=L)
        R = cap[f"resid_pre{L}"]
        x = cap[f"xn{L}"][rows, tpos]  # [B, 768]
    st = (L, R)

    def y_of(**kw):
        with torch.no_grad():
            lg, _ = m.forward(ids, start=st, read_pos=(rows, tpos), **kw)
        return centred(lg, nxt)

    def grad(lin):
        dv = torch.zeros(B, 768, device=DEV, requires_grad=True)
        lg, _ = m.forward(ids, start=st, mlp_out_delta={L: (rows, tpos, dv)}, linearise=lin, read_pos=(rows, tpos))
        centred(lg, nxt).sum().backward()
        return dv.grad.detach()

    gg = grad(L); gr = grad(None)
    with torch.no_grad():
        y0 = y_of()
        # unit pre-activation: feature a_f = (x - b_dec).e_f + b_enc_f; oracle z_n = (x g2 + b2).W_fc[:,n] + b_fc[n]
        uid = torch.tensor([units[k][2] for k in c[:, 0]], device=DEV)
        pf = ((x - tc.b_dec) * e).sum(-1) + tc.b_enc[uid.clamp(max=F - 1)]
        pn = ((x * g2 + b["ln_2.bias"]) * Wfc[:, uid.clamp(max=3071)].T).sum(-1) + bfc[uid.clamp(max=3071)]
        pre = torch.where(orc, pn, pf)
        en2 = (e * e).sum(-1, keepdim=True)
        du = -pre[:, None] * e / en2
        dun = du.norm(dim=-1, keepdim=True)
        r = torch.randn(B, 768, generator=gen).to(DEV); r = r / r.norm(dim=-1, keepdim=True)
        du_r = r * dun
        dr = torch.randn(B, 768, generator=gen).to(DEV); dr = dr - dr.mean(-1, keepdim=True)
        dr = dr / dr.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
        m0, t0_ = mlp_c(x), tc(x)
        dm = mlp_c(x + du) - m0; dt = tc(x + du) - t0_
        dm_mir = mlp_c(x - du) - m0; dt_mir = tc(x - du) - t0_
        dm_r = mlp_c(x + du_r) - m0; dt_r = tc(x + du_r) - t0_
        act = (tc.pre(x) > 0).float()
        flips = ((tc.pre(x + du) > 0).float() != act).sum(-1).float()  # incl. the unit itself for features
        flips_r = ((tc.pre(x + du_r) > 0).float() != act).sum(-1).float()
        zc = zpre(x)
        gpz = gelu_new_grad(zc)
        Jdu = (gpz * ((du * g2) @ Wfc)) @ Wpc
        Jdu_r = (gpz * ((du_r * g2) @ Wfc)) @ Wpc
        jv = (act * (du @ tc.W_enc)) @ tc.W_dec
        gelu_z = gelu_new(pre)
        a_node = torch.where(orc, gelu_z - gelu_new(torch.zeros_like(pre)), pre)  # size of the unit's own term
        node = -a_node[:, None] * d
        floor = -a_node[:, None] * dr
        dot = lambda u, v: (u * v).sum(-1)
        out_b = {"pre_ok": (pre > 0).float(), "pre": pre, "du_norm": dun.squeeze(1), "xn_norm": x.norm(dim=-1)}
        for gk, gv in [("g", gg), ("r", gr)]:
            out_b[f"meas_local_{gk}"] = dot(dm, gv); out_b[f"total_{gk}"] = dot(dt, gv)
            out_b[f"jvp_{gk}"] = dot(jv, gv); out_b[f"fonly_{gk}"] = dot(node, gv); out_b[f"floor_{gk}"] = dot(floor, gv)
        for gk, gv in [("g", gg), ("r", gr)]:
            out_b[f"mir_meas_{gk}"] = dot(dm_mir, gv); out_b[f"mir_total_{gk}"] = dot(dt_mir, gv)
            out_b[f"rnd_meas_{gk}"] = dot(dm_r, gv); out_b[f"rnd_total_{gk}"] = dot(dt_r, gv)
        out_b["g_cos"] = torch.nn.functional.cosine_similarity(gg, gr, dim=-1)
        out_b["g_norm_ratio"] = gg.norm(dim=-1) / gr.norm(dim=-1)
        out_b["cos_vec"] = torch.nn.functional.cosine_similarity(dm, dt, dim=-1)
        out_b["tr2"] = ((dt - dm) ** 2).sum(-1); out_b["r2"] = (dm ** 2).sum(-1)  # Prediction C
        out_b["tr2_rnd"] = ((dt_r - dm_r) ** 2).sum(-1); out_b["r2_rnd"] = (dm_r ** 2).sum(-1)
        out_b["flips"] = flips; out_b["flips_rnd"] = flips_r
        out_b["curv"] = (dm - Jdu).norm(dim=-1) / Jdu.norm(dim=-1)  # Iris H2
        out_b["curv_rnd"] = (dm_r - Jdu_r).norm(dim=-1) / Jdu_r.norm(dim=-1)
        # rank-1 alignment: J = W_out_c^T diag(gelu'(z)) W_fc^T diag(g2)  (x_n -> centred out)
        z = zpre(x)
        gp = gelu_new_grad(z)
        # |J|_F^2 = gp^T (P P^T o Q Q^T) gp with P = W_out_c, Q[n] = g2 * W_fc[:, n]
        Jfro = ((gp @ KJ) * gp).sum(-1).clamp(min=0).sqrt()
        dJe = ((d @ Wpc.T) * gp * ((e * g2) @ Wfc)).sum(-1)
        out_b["rank1"] = dJe / (d.norm(dim=-1) * e.norm(dim=-1) * Jfro)
        # end-to-end forwards
        out_b["e2e_real"] = y_of(mlp_in_delta={L: (rows, tpos, du)}) - y0
        out_b["e2e_tc"] = y_of(mlp_out_delta={L: (rows, tpos, dt)}) - y0
        out_b["e2e_rnd"] = y_of(mlp_in_delta={L: (rows, tpos, du_r)}) - y0
        # WRITE: MLP_L out at t := TC(x) (+/- a d)
        mraw = m.mlp(b, x)
        base = t0_ - mraw
        yb = y_of(mlp_out_delta={L: (rows, tpos, base)})
        ad = a_node[:, None] * d
        rd = dr * a_node[:, None]
        out_b["wr_minus"] = y_of(mlp_out_delta={L: (rows, tpos, base - ad)}) - yb
        out_b["wr_plus"] = y_of(mlp_out_delta={L: (rows, tpos, base + ad)}) - yb
        out_b["wr_rnd_minus"] = y_of(mlp_out_delta={L: (rows, tpos, base - rd)}) - yb
        out_b["wr_rnd_plus"] = y_of(mlp_out_delta={L: (rows, tpos, base + rd)}) - yb
        out_b["wr_pred"] = dot(ad, gg); out_b["wr_rnd_pred"] = dot(rd, gg)
        out_b["wr_pred_r"] = dot(ad, gr); out_b["wr_rnd_pred_r"] = dot(rd, gr)
        out_b["y_clean"] = y0
        out_b["y_full_minus_cache"] = centred(lg_full, nxt) - y0
    for k, v in out_b.items():
        out[k][sl] = v.float().cpu().double().numpy()
    if bi % 10 == 0:
        print(f"[{tag}] {s+B}/{len(ctx)} {time.time()-t0:.0f}s fp {fp()}", flush=True)
    torch.mps.empty_cache()

# ---------------------------------------------------------------- save
import csv

with open(f"data/pushes/{tag}_units.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["L", "group", "seed", "uid", "enc_cos", "dec_cos", "joint", "same_neuron", "max_enc_any",
                 "max_dec_any", "e_norm", "d_norm", "n_active", "fire_rate", "mean_act"])
    for k, (gname, s, u) in enumerate(units):
        if gname == "ORACLE":
            row = [L, gname, s, u, 1, 1, 1, u, 1, 1, float(E_units[k].norm()), float(D_units[k].norm()),
                   int(ncnt[u]), float(ncnt[u] / ntok), ""]
        else:
            row = [L, gname, s, u, float(enc_cos[u]), float(dec_cos[u]), float(joint[u]), int(nstar[u]),
                   float(max_enc[u]), float(max_dec[u]), float(tc.W_enc[:, u].norm()), float(tc.W_dec[u].norm()),
                   int(cnt[u]), float(cnt[u] / ntok), float(sact[u] / max(cnt[u], 1))]
        wr.writerow(row)
cols = list(out)
with open(f"data/pushes/{tag}_pushes.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["unit", "group", "seed", "uid", "window", "t"] + cols)
    for i, (k, w, t) in enumerate(ctx):
        gname, s, u = units[k]
        wr.writerow([k, gname, s, u, w, t] + [f"{out[c][i]:.7g}" for c in cols])
meta = {"layer": L, "args": vars(args), "ntok": ntok, "n_units": len(units), "n_pushes": len(ctx),
        "counts": {"NEURON_all": int(NEU.sum()), "NEURON_active": int((NEU & alive).sum()),
                   "REST_all": int(RST.sum()), "REST_active": int((RST & alive).sum()),
                   "GAP_active_bins": [int(len(p)) for p in gap_bins], "neurons_active": int((ncnt > 0).sum())},
        "gap_short_bins": gap_short, "seconds": time.time() - t0, "columns": COLUMNS}
json.dump(meta, open(f"data/pushes/{tag}_meta.json", "w"), indent=1)
print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)

"""Phase 0 (PREREG s2) + the frozen null sets (PREREG s3) for switch-slow-band. Base (untrained) 2-block tail of
Qwen3-0.6B, monitor switch inputs (cue-on = ref, cue-off = none), 200 test prompts, CPU fp32. No patch rows.

Outputs
  data/sets.json   U1..U50 (rng 1103, uniform 3-sets of 0..63, != BAND), N1..N200 (rng 2860, 3-sets of 28..60),
                   E1..E20 (rng 7, 3-sets of 0..60 with sum e within +-10% of BAND's; e_p below), TG (all 3-sets of
                   the top-12 block-27 k_norm-gain pairs minus {61,62,63}). Sets are sorted and distinct within a null.
  data/phase0.json pooled sink cut (arm Z: pairs 61-63 zeroed in q.k in blocks 26 and 27, one pass, cue-on input),
                   bootstrap CI, U and N cuts, per-pair single-zero cuts, per-head answer-position token-0 mass with
                   the band zeroed in that head's own layer only, the head-27.8 test, the per-pair token-0 lead
                   split G (all 32 heads; head 27.8 = G27.8), e_p, k_norm gains.
Usage: env/.venv/bin/python phase0.py [--threads 6]
"""
import argparse, itertools, json, os, time
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
os.environ["NC_DEV"] = "cpu"
import numpy as np, torch
torch.set_num_threads(a.threads)
import pp, patch as P
from transformers.integrations.sdpa_attention import repeat_kv

t0 = time.time()
BAND = [61, 62, 63]
R = pp.Runner("fp32").load("monitor_s0", base=True)
tail = R.tail
OUT = pp.C.HERE / "data"; OUT.mkdir(exist_ok=True)


@torch.no_grad()
def cap(u16):
    """q [B,16,T,128], k [B,16,T,128] (post k_norm + RoPE, repeated to query heads) per block, fp32."""
    qs, ks = {0: [], 1: []}, {0: [], 1: []}
    for _, h in R.batches(u16):
        P.CTRL.update(mode="capture"); tail(h, R.w_out)
        for li in range(2):
            qs[li].append(P.CAP[f"q{li}"].float()); ks[li].append(repeat_kv(P.CAP[li][0], 2).float())
    P.CTRL.update(mode="none")
    return {li: torch.cat(qs[li]) for li in range(2)}, {li: torch.cat(ks[li]) for li in range(2)}


# ---- e_p (cue-independent energy, PREREG s3 E): block 27, answer query, cue-off input, base tail
q_off, k_off = cap(R.off)
qa = q_off[1][:, :, -1].view(-1, 16, 2, 64).pow(2).sum(2).sqrt()                  # [B,16,64]
kk = k_off[1][:, :, 1:3].reshape(-1, 16, 2, 2, 64).pow(2).sum(3).sqrt().mean(2)   # [B,16,64] mean over t=1,2
e = (qa * kk).mean((0, 1)).double().numpy()
eB = e[BAND].sum()

# ---- k_norm gains (block 27, base weights)
w = tail.layers[1].self_attn.k_norm.weight.detach().double().cpu().numpy()
gain = np.sqrt(w[:64] ** 2 + w[64:] ** 2)
top12 = [int(i) for i in np.argsort(-gain, kind="stable")[:12]]

# ---- null sets
def draw(rng, pool, n, ok=lambda s: True, max_draws=None):
    out, seen, k = [], set(), 0
    while len(out) < n and (max_draws is None or k < max_draws):
        s = tuple(sorted(int(x) for x in rng.choice(pool, 3, replace=False))); k += 1
        if s in seen or list(s) == BAND or not ok(s):
            continue
        seen.add(s); out.append(list(s))
    return out, k

U, _ = draw(np.random.default_rng(1103), np.arange(64), 50)
N, _ = draw(np.random.default_rng(2860), np.arange(28, 61), 200)
E, nE = draw(np.random.default_rng(7), np.arange(61), 20, ok=lambda s: abs(e[list(s)].sum() - eB) <= 0.1 * eB,
             max_draws=100000)
E_short = 20 - len(E)
if E_short:
    allc = sorted(itertools.combinations(range(61), 3), key=lambda s: abs(e[list(s)].sum() - eB))
    E += [list(s) for s in allc if list(s) not in E][:E_short]
rest = [p for p in top12 if p not in BAND]
TG = [list(s) for s in itertools.combinations(sorted(rest), 3)]
sets = {"BAND": BAND, "U": U, "N": N, "E": E, "TG": TG,
        "rule": "PREREG v1 s3. U: default_rng(1103).choice(arange(64),3,replace=False) sorted, skip ==BAND or repeat, "
                "first 50. N: default_rng(2860) over arange(28,61), first 200 distinct. E: default_rng(7) over "
                "arange(61), keep sets with |sum e - sum e_BAND| <= 0.1 sum e_BAND, first 20 distinct within 1e5 draws; "
                "shortfall filled by nearest-energy sets of all C(61,3). TG: all 3-sets of top-12 block-27 k_norm "
                "gain pairs minus BAND.",
        "E_draws": nE, "E_shortfall_filled": E_short, "e": e.tolist(), "e_BAND_sum": float(eB),
        "E_sum_e": [float(e[s].sum()) for s in E], "kgain27": gain.tolist(), "top12_gain27": top12,
        "band_in_top12": [p for p in BAND if p in top12]}
json.dump(sets, open(OUT / "sets.json", "w"), indent=1)
print("sets written", len(U), len(N), len(E), "(shortfall", E_short, ")", len(TG), "top12", top12, flush=True)

# ---- G: exact per-pair token-0 lead split at the answer position, cue-on input, base tail, all 32 heads
q_on, k_on = cap(R.on)
G, lead_err = {}, 0.0
for li in range(2):
    q = q_on[li][:, :, -1]                                   # [B,16,128]
    k = k_on[li]                                             # [B,16,T,128]
    kbar = k[:, :, 1:].mean(2)                               # mean over keys 1..33 (answer query sees all)
    c = (q * (k[:, :, 0] - kbar)).view(-1, 16, 2, 64).sum(2) / np.sqrt(128)     # [B,16,64]
    lg = (q[:, :, None] * k).sum(-1) / np.sqrt(128)           # [B,16,T]
    lead = lg[:, :, 0] - lg[:, :, 1:].mean(-1)
    lead_err = max(lead_err, float((c.sum(-1) - lead).abs().max()))
    G[26 + li] = c.mean(0).double().numpy()                   # [16,64]
assert lead_err < 1e-4, lead_err
g278 = G[27][8]

# ---- Phase 0 sink arms (forward with zeroing; cue-on input)
@torch.no_grad()
def zrun(z26, z27):
    P.CTRL.update(mode="zero", zset=[z26, z27], zrec={0: [], 1: []})
    for _, h in R.batches(R.on):
        tail(h, R.w_out)
    rec = P.CTRL["zrec"]; P.CTRL.update(mode="none")
    t0_ = torch.cat([torch.cat([r["t0"] for r in rec[li]]) for li in range(2)], 1).double().numpy()   # [B,32]
    ans = torch.cat([torch.cat([r["ans"] for r in rec[li]]) for li in range(2)], 1).double().numpy()  # [B,32,3]
    return t0_, ans

base_t0, base_ans = zrun(None, None)
Sb = base_t0.mean()
def cut(t0_):
    return float(1 - t0_.mean() / Sb)
zb_t0, zb_ans = zrun(BAND, BAND)
rng = np.random.default_rng(0)
nP = base_t0.shape[0]
bs = []
for _ in range(2000):
    i = rng.integers(0, nP, nP)
    bs.append(1 - zb_t0[i].mean() / base_t0[i].mean())
res = {"S_base": float(Sb), "S_Z": float(zb_t0.mean()), "cut_Z": cut(zb_t0),
       "cut_Z_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
res["U_cuts"] = [cut(zrun(s, s)[0]) for s in U]
res["N_cuts"] = [cut(zrun(s, s)[0]) for s in N]
res["single_cuts"] = [cut(zrun([j], [j])[0]) for j in range(64)]
print("phase0 cut", res["cut_Z"], res["cut_Z_ci95"], "U median", np.median(res["U_cuts"]), "N95",
      np.percentile(res["N_cuts"], 95), f"{time.time() - t0:.0f}s", flush=True)
# per head, band zeroed in that head's own layer only
z26_t0, z26_ans = zrun(BAND, None)
z27_t0, z27_ans = zrun(None, BAND)
own_ans = np.concatenate([z26_ans[:, :16], z27_ans[:, 16:]], 1)       # [B,32,3]
own_t0 = np.concatenate([z26_t0[:, :16], z27_t0[:, 16:]], 1)
heads = {}
for hh in range(32):
    b = float(base_ans[:, hh, 0].mean()); z = float(own_ans[:, hh, 0].mean())
    heads[f"{26 + hh // 16}.{hh % 16}"] = {"ans_t0_base": b, "ans_t0_Z": z, "ans_cut": (1 - z / b) if b > 0 else None,
                                           "ans_t12_base": float(base_ans[:, hh, 1:].sum(-1).mean()),
                                           "t0_base": float(base_t0[:, hh].mean()), "t0_Z": float(own_t0[:, hh].mean())}
h = heads["27.8"]
res["head_27_8"] = dict(h, slow_band_sink_head=bool(h["ans_t0_base"] >= 0.5 and h["ans_cut"] >= 0.5))
res["per_head"] = heads
res["n_heads_t0_base_ge_0.5"] = int(sum(v["t0_base"] >= 0.5 for v in heads.values()))
res["G"] = {str(k): v.tolist() for k, v in G.items()}
res["G27_8"] = g278.tolist()
res["G27_8_top3"] = [int(i) for i in np.argsort(-g278)[:3]]
res["G27_8_band_share"] = float(g278[BAND].sum() / g278.sum())
res["G27_8_band_share_of_positive"] = float(g278[BAND].clip(0).sum() / g278.clip(0).sum())
res["lead_split_max_err"] = lead_err
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(OUT / "phase0.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_head", "G", "G27_8", "U_cuts", "N_cuts", "single_cuts")}, indent=1))
print("single-pair cut top5", np.argsort(-np.array(res["single_cuts"]))[:5].tolist())

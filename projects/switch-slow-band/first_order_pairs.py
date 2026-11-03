"""Frozen prediction (PREREG s4), written and run BEFORE any patch row. Owner: Iris (research).

It reads only queries and keys of the unpatched runs: no key is swapped, no P(X) of a patched run is computed.
Per monitor copy (fp32, CPU), from one capture pass on the cue-on input and one on the cue-off input:
  q    = cue-on query at the answer position (last token, index 33), blocks 26 and 27, all 16 query heads
  k_on, k_off = post-k_norm, post-RoPE keys at tokens 0..33 (cue-on) and 0..2 (cue-off), KV head = query head // 2
Exact per-pair score split (an identity; asserted):
  ds_{h,j}(p) = scaling * q_p . (k_off,p(j) - k_on,p(j)),  pair p = dims (p, p+64),  j in {0, 1, 2}
Predicted attention of head 27.8 for a pair set S (block-27 queries held at the cue-on run):
  a_S = softmax_j( s_on(j) + sum_{p in S} ds(p)[j] ),   per prompt, then averaged over the 200 prompts
  r_S = (a_S(t2) - a_on(t2)) / (a_all(t2) - a_on(t2))
  pred_S = r_S * share_G      (share_G = this project's gate row if it exists, else sink-value fp32 CPU K-only)
Reading predictor (not in the prereg status): first-order linear share from d P / d alpha_p at alpha = 0
(pp.Runner.grad_alpha: autograd through the cue-on run, no finite patch), lin_S = sum_{p in S} g_p / (P_off - P_on).

Also written: the BAND share of the head-27.8 (t1 - t2) score-gap change (identity), the per-pair ||dk||^2 share,
base-tail energies e_p for the E null, and the U / N / E pair sets (drawn here by the PREREG seeds when
data/sets.json does not exist yet; Bruno's runner loads the same file).

    NC_DEV=cpu env/.venv/bin/python kran-research/projects/switch-slow-band/first_order_pairs.py
"""
import json
import os
import time

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import patch as P  # noqa: E402
import pp  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
HERE = pp.C.HERE
NPAIR, ANS, L, H = 64, 33, 1, 8          # block index 1 = block 27; head 8 = carrier head 27.8
BAND = [61, 62, 63]
COMP = [p for p in range(NPAIR) if p not in BAND]
SHARE_K_SINKVALUE = {"monitor_s0": 0.98384, "monitor_s1": 0.98611}   # sink-value fp32_cpu K-only (PREREG s4)
t0 = time.time()


def capture_qk(R, u16):
    """[B,16,T,128] cue-on queries and [B,8,T,128] keys for both blocks (post norm + RoPE), fp32 CPU."""
    qs, ks = {0: [], 1: []}, {0: [], 1: []}
    with torch.no_grad():
        for _, h in R.batches(u16):
            P.CTRL.update(mode="capture")
            R.tail(h, R.w_out)
            for i in range(2):
                qs[i].append(P.CAP[f"q{i}"].float().cpu())
                ks[i].append(P.CAP[i][0].float().cpu())
    P.CTRL.update(mode="none")
    return {i: torch.cat(qs[i]).double() for i in range(2)}, {i: torch.cat(ks[i]).double() for i in range(2)}


def draw_sets(e):
    """U: uniform 3-sets 0..63 (rng 1103, != BAND); N: 3-sets from 28..60 (rng 2860); E: energy-matched 3-sets from
    0..60, sum e within +-10% of BAND (rng 7, <= 1e5 draws, first 20; shortfall filled by nearest energy)."""
    rU = np.random.default_rng(1103)
    U = []
    while len(U) < 50:
        s = sorted(rU.choice(NPAIR, 3, replace=False).tolist())
        if s != BAND:
            U.append(s)
    rN = np.random.default_rng(2860)
    N = [sorted(rN.choice(np.arange(28, 61), 3, replace=False).tolist()) for _ in range(200)]
    rE = np.random.default_rng(7)
    eb = float(e[BAND].sum())
    E, near, n_draws = [], [], 0
    while len(E) < 20 and n_draws < 100_000:
        s = sorted(rE.choice(61, 3, replace=False).tolist())
        n_draws += 1
        es = float(e[s].sum())
        if abs(es - eb) <= 0.1 * eb:
            E.append(s)
        else:
            near.append((abs(es - eb), s))
    found = len(E)
    if found < 20:
        seen = {tuple(s) for s in E}
        for _, s in sorted(near):
            if tuple(s) not in seen:
                E.append(s)
                seen.add(tuple(s))
            if len(E) == 20:
                break
    return {"U": U, "N": N, "E": E, "E_found_within_10pct": found, "E_draws": n_draws, "e_band": eb,
            "rules": "PREREG v1 s3; drawn by first_order_pairs.py"}


R = pp.Runner("fp32")
assert pp.T.DEV == "cpu"
sets_path = HERE / "data/sets.json"
out = {"note": "frozen prediction, PREREG s4; no patch row had run when this file was written",
       "band": BAND, "copies": {}}

# Base tail energies for the E null (cue-independent): e_p = mean over prompts and the 16 block-27 query heads of
# |q_p(answer)| x mean_{t in 1,2} |k_p(t)|, base tail, cue-off input.
R.load("monitor_s0", base=True)
qb, kb = capture_qk(R, R.off)
q27 = qb[L][:, :, ANS].reshape(-1, 16, 2, NPAIR).pow(2).sum(2).sqrt()                # [B,16,64]
k27 = kb[L][:, :, 1:3].reshape(-1, 8, 2, 2, NPAIR).pow(2).sum(3).sqrt().mean(2)       # [B,8,64]
k27 = k27.repeat_interleave(2, dim=1)                                                 # KV head g -> q heads 2g, 2g+1
e = (q27 * k27).mean((0, 1)).numpy()


def sink_split(qd, kd, qpos):
    """Exact per-pair token-0 lead (rope-sink-band rsb.py:77 / PREREG v1.1, Kasper #13490): for each query head of
    both blocks, c_j = scaling * [q_j(t).k_j(0) - mean_{u=1..t} q_j(t).k_j(u)], averaged over prompts and the query
    positions qpos. Returns {block: [16,64]}. sum_j c_j = the token-0 score lead over the mean other key."""
    res = {}
    for b in range(2):
        acc = torch.zeros(16, NPAIR, dtype=torch.float64)
        n = qd[b].shape[0]
        for s in range(0, n, 20):                                          # prompt chunks keep RSS < 0.3 GB
            q = qd[b][s:s + 20][:, :, qpos].reshape(-1, 16, len(qpos), 2, NPAIR)
            k = kd[b][s:s + 20].repeat_interleave(2, dim=1)
            k = k.reshape(k.shape[0], 16, -1, 2, NPAIR)
            pp_ = torch.einsum("bhqcp,bhtcp->bhqtp", q, k) * 128 ** -0.5   # [b,16,Q,T,64]
            for qi, t in enumerate(qpos):
                acc += (pp_[:, :, qi, 0] - pp_[:, :, qi, 1:t + 1].mean(2)).sum(0)
        res[26 + b] = acc / (n * len(qpos))                                # [16,64]
    return res


# Head 27.8's own sink profile on THESE prompts, BASE tail, cue-on and cue-off inputs (tokens 0-2 differ only at 1-2).
out["sink_profile_base"] = {}
for cond in ("on", "off"):
    qd, kd = capture_qk(R, R.on if cond == "on" else R.off)
    for tag, qpos in (("answer", [ANS]), ("mean_t3_33", list(range(3, 34)))):
        sp = sink_split(qd, kd, qpos)
        g8 = sp[27][H].numpy()
        top3 = [int(p) for p in np.argsort(-g8)[:3]]
        a0 = torch.softmax((qd[L][:, H, ANS:ANS + 1] @ kd[L][:, H // 2].transpose(-1, -2))[:, 0] * 128 ** -0.5, -1)
        out["sink_profile_base"][f"{cond}_{tag}"] = {
            "head27_8_per_pair": g8.round(6).tolist(), "head27_8_total_lead": float(g8.sum()),
            "head27_8_band_share": float(g8[BAND].sum() / g8.sum()), "head27_8_top3": top3,
            "head27_8_answer_att_t0": float(a0[:, 0].mean()),
            "all_heads_band_share": {f"{blk}.{h}": round(float(sp[blk][h, BAND].sum() / sp[blk][h].sum()), 4)
                                     for blk in (26, 27) for h in range(16)},
            "all_heads_per_pair": {str(blk): sp[blk].numpy().round(5).tolist() for blk in (26, 27)}}
        print("base sink profile", cond, tag, "head 27.8 band share",
              round(out["sink_profile_base"][f"{cond}_{tag}"]["head27_8_band_share"], 4), "top3", top3, flush=True)
out["base_energy_e"] = e.round(6).tolist()
out["base_energy_rank_of_band"] = [int(np.where(np.argsort(-e) == p)[0][0]) for p in BAND]
if sets_path.exists():
    sets = json.load(open(sets_path))
    out["sets_source"] = "data/sets.json (existing)"
else:
    sets = draw_sets(e)
    json.dump(sets, open(sets_path, "w"), indent=1)
    out["sets_source"] = "data/sets.json (drawn here)"
print("base energies done", round(time.time() - t0, 1), "s; band energy ranks", out["base_energy_rank_of_band"],
      flush=True)

gate_rows = {}
for name in ("monitor_s0", "monitor_s1"):
    gp = HERE / f"data/gate_{name}_fp32_cpu.json"
    if gp.exists():
        gate_rows[name] = json.load(open(gp)).get("K")

for name in ("monitor_s0", "monitor_s1"):
    R.load(name)
    qon, kon = capture_qk(R, R.on)
    _, koff = capture_qk(R, R.off)
    scal = R.tail.layers[L].self_attn.scaling
    assert abs(scal - 128 ** -0.5) < 1e-9
    c = {"P_on": R.P_on, "P_off": R.P_off}
    # ---- exact per-pair split, every head of both blocks, tokens 0..2 ----
    per_head_gap = {}
    max_err = 0.0
    for b in range(2):
        q = qon[b][:, :, ANS]                                            # [B,16,128]
        dk = (koff[b][:, :, :3] - kon[b][:, :, :3]).repeat_interleave(2, dim=1)   # [B,16,3,128]
        prod = q[:, :, None, :] * dk * scal                               # [B,16,3,128]
        ds = prod.reshape(*prod.shape[:3], 2, NPAIR).sum(3)               # [B,16,3,64]
        full = (q[:, :, None, :] * dk).sum(-1) * scal
        max_err = max(max_err, float((ds.sum(-1) - full).abs().max()))
        gap = (ds[:, :, 1] - ds[:, :, 2]).mean(0)                          # [16,64] change of (t1 - t2) score gap
        per_head_gap[26 + b] = gap
        if b == L:
            ds27 = ds
    assert max_err < 1e-4, max_err
    c["split_identity_max_abs_err"] = max_err
    g = per_head_gap[27][H].numpy()
    c["head27_8_gap_change_per_pair"] = g.round(6).tolist()
    c["head27_8_gap_change_total"] = float(g.sum())
    c["head27_8_gap_band_share"] = float(g[BAND].sum() / g.sum())
    dk8 = (koff[L][:, H // 2, 1:3] - kon[L][:, H // 2, 1:3])               # [B,2,128]
    en = dk8.reshape(-1, 2, 2, NPAIR).pow(2).sum(2).mean((0, 1)).numpy()
    c["head27_8_dk_energy_band_share"] = float(en[BAND].sum() / en.sum())
    c["head27_8_dk_energy_per_pair"] = (en / en.sum()).round(6).tolist()
    # ---- head-27.8 softmax prediction ----
    q8 = qon[L][:, H, ANS]                                                 # [B,128]
    s_on = (kon[L][:, H // 2] @ q8[:, :, None])[..., 0] * scal              # [B,34] scores, keys 0..33
    a_on = torch.softmax(s_on, -1)

    def att(S):
        s = s_on.clone()
        if len(S):
            s[:, :3] = s[:, :3] + ds27[:, H, :, S].sum(-1)
        return torch.softmax(s, -1)

    a_all = att(list(range(NPAIR)))
    c["head27_8_att_on"] = a_on[:, :3].mean(0).numpy().round(5).tolist()
    c["head27_8_att_allpairs_q_held"] = a_all[:, :3].mean(0).numpy().round(5).tolist()
    den = float(a_all[:, 2].mean() - a_on[:, 2].mean())
    share_G = gate_rows.get(name)
    c["share_G_used"] = share_G if share_G is not None else SHARE_K_SINKVALUE[name]
    c["share_G_source"] = "gate row (this project)" if share_G is not None else "sink-value fp32_cpu K-only"

    def r(S):
        return float(att(S)[:, 2].mean() - a_on[:, 2].mean()) / den

    # ---- first-order linear reading predictor ----
    gr = R.grad_alpha()
    lin = gr / (R.P_off - R.P_on)
    c["grad_alpha_share_per_pair"] = lin.round(6).tolist()
    c["grad_alpha_sum_share"] = float(lin.sum())

    rows = {"BAND": BAND, "COMP": COMP}
    rows.update({f"S{p}": [p] for p in range(NPAIR)})
    rows.update({f"U{i + 1}": s for i, s in enumerate(sets["U"])})
    rows.update({f"N{i + 1}": s for i, s in enumerate(sets["N"])})
    rows.update({f"E{i + 1}": s for i, s in enumerate(sets["E"])})
    rows.update({"LOW8": list(range(56, 64)), "SLOW36": list(range(28, 64))})
    pred = {}
    for k, S in rows.items():
        rs = r(S)
        pred[k] = {"pairs": S, "r": round(rs, 6), "pred": round(rs * c["share_G_used"], 6),
                   "lin": round(float(lin[S].sum()), 6)}
    c["pred"] = pred
    sumN = sorted(v["pred"] for k, v in pred.items() if k.startswith("N"))
    sumU = sorted(v["pred"] for k, v in pred.items() if k.startswith("U"))
    sumE = sorted(v["pred"] for k, v in pred.items() if k.startswith("E"))
    c["summary"] = {"pred_BAND": pred["BAND"]["pred"], "pred_COMP": pred["COMP"]["pred"],
                    "pred_N95": float(np.percentile(sumN, 95)), "pred_U_median": float(np.median(sumU)),
                    "pred_E_median": float(np.median(sumE)), "lin_BAND": pred["BAND"]["lin"],
                    "top5_single_pred": sorted(((pred[f"S{p}"]["pred"], p) for p in range(NPAIR)), reverse=True)[:5],
                    "top5_single_lin": sorted(((float(lin[p]), p) for p in range(NPAIR)), reverse=True)[:5]}
    # Law row, prediction side (reading): rank correlation over the 64 pairs between head 27.8's own base sink lead
    # (answer position, cue-on) and the pair's push of the score gap toward token 2 (-gap change) and pred_S.
    sink8 = np.array(out["sink_profile_base"]["on_answer"]["head27_8_per_pair"])
    predS = np.array([pred[f"S{p}"]["pred"] for p in range(NPAIR)])

    def spearman(a, b_, nperm=10_000):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b_))
        rho = float(np.corrcoef(ra, rb)[0, 1])
        rng = np.random.default_rng(0)
        null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(nperm)])
        return rho, float((np.sum(null >= rho) + 1) / (nperm + 1))

    c["law_pred"] = {"spearman_sink8_vs_minus_gap": spearman(sink8, -g), "spearman_sink8_vs_predS": spearman(sink8, predS)}
    out["copies"][name] = c
    print(name, json.dumps(c["summary"]), "gap band share", round(c["head27_8_gap_band_share"], 4),
          "dk energy band share", round(c["head27_8_dk_energy_band_share"], 4), round(time.time() - t0, 1), "s",
          flush=True)

out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(HERE / "data/prediction.json", "w"), indent=1)
print("wrote data/prediction.json")

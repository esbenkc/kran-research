"""EXPLORATORY, post-data (claimed after the status rows, room #13537). Owner: Iris (research).

Why the frozen prediction (data/prediction.json) got the per-pair shape right in monitor s0 but the scale ~5x too
big: it assumed P(X) moves in proportion to head 27.8's attention on token 2. This script measures that map.

Per monitor copy, fp32 CPU:
  1. Whole-key alpha path: key = k_on + alpha (k_off - k_on) in every pair, tokens 0-2, all 32 heads of blocks 26-27
     (the same arm as row G at alpha = 1), alpha = 0, 0.05, ..., 1. P_i(alpha) per prompt i (forward passes).
  2. Head 27.8's token-2 attention along the same path, a_i(alpha), from the exact q-held softmax (prediction.json
     showed the q-held all-pair attention equals the real cue-off attention to 1e-5).
  3. For every pair set S of the status rows: a_i(S) = q-held softmax with only S's score terms; the readout-corrected
     prediction P_i(S) = P_i at the alpha where a_i(alpha) = a_i(S) (linear interpolation on the grid, per prompt);
     pred2_S = (mean_i P_i(S) - P_on) / (P_off - P_on). Compared with the measured share of S (data/patch/*.json).
Output: data/readout_curve.json.

    NC_DEV=cpu env/.venv/bin/python kran-research/projects/switch-slow-band/readout_curve.py
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
NPAIR, ANS, L, H = 64, 33, 1, 8
ALPHAS = np.round(np.linspace(0, 1, 21), 4)
t0 = time.time()


def capture_qk(R, u16):
    qs, ks = [], []
    with torch.no_grad():
        for _, h in R.batches(u16):
            P.CTRL.update(mode="capture")
            R.tail(h, R.w_out)
            qs.append(P.CAP[f"q{L}"][:, H, ANS].double().cpu())
            ks.append(P.CAP[L][0][:, H // 2].double().cpu())
    P.CTRL.update(mode="none")
    return torch.cat(qs), torch.cat(ks)                         # [B,128], [B,34,128]


R = pp.Runner("fp32")
assert pp.T.DEV == "cpu"
out = {"note": "EXPLORATORY, post-data (room #13537); not a status row", "alphas": ALPHAS.tolist(), "copies": {}}
for name in ("monitor_s0", "monitor_s1"):
    R.load(name)
    q, kon = capture_qk(R, R.on)
    _, koff = capture_qk(R, R.off)
    scal = 128 ** -0.5
    s_on = (kon @ q[:, :, None])[..., 0] * scal                    # [B,34]
    dk = koff[:, :3] - kon[:, :3]                                    # [B,3,128]
    ds = (q[:, None, :] * dk * scal).reshape(-1, 3, 2, NPAIR).sum(2)  # [B,3,64]

    def att2(w):                                                     # w: [64] weights on each pair's score term
        s = s_on.clone()
        s[:, :3] = s[:, :3] + (ds * torch.as_tensor(w, dtype=torch.float64)).sum(-1)
        return torch.softmax(s, -1)[:, 2].numpy()

    A = np.stack([att2(np.full(NPAIR, a)) for a in ALPHAS])          # [21,B] token-2 attention on the path
    Pp = []
    for a in ALPHAS:
        spec = P.arm(pp.ALL16, pp.ALL16, kpos=pp.F3, alpha=torch.full((NPAIR,), float(a)).to(pp.T.DEV))
        Pp.append(R.px(R.on, spec))
    Pp = np.stack(Pp)                                                # [21,B]
    assert abs(Pp[0].mean() - R.P_on) < 1e-9
    meas = json.load(open(HERE / f"data/patch/{name}_fp32_cpu.json"))
    assert abs(meas["P_on"] - R.P_on) < 1e-9 and abs(meas["P_off"] - R.P_off) < 1e-9
    share_path = (Pp.mean(1) - R.P_on) / (R.P_off - R.P_on)
    att_path = (A.mean(1) - A[0].mean()) / (A[-1].mean() - A[0].mean())
    mono = float(np.mean(np.all(np.diff(A, axis=0) >= -1e-12, axis=0)))

    def pred2(S):
        w = np.zeros(NPAIR)
        w[list(S)] = 1.0
        aS = att2(w)
        Pi = np.array([np.interp(aS[i], A[:, i], Pp[:, i]) if A[-1, i] >= A[0, i]
                       else np.interp(-aS[i], -A[:, i], Pp[:, i]) for i in range(len(aS))])
        return float((Pi.mean() - R.P_on) / (R.P_off - R.P_on)), float((aS.mean() - A[0].mean()) / (A[-1].mean() - A[0].mean()))

    rows = {}
    for k, v in meas["rows"].items():
        if k == "G" or "pairs" not in v or not v["pairs"]:
            continue
        p2, r = pred2(v["pairs"])
        rows[k] = {"pairs": v["pairs"], "measured": v["share"], "pred2_readout": round(p2, 6), "att_shift": round(r, 6)}
    singles = [k for k in rows if k.startswith("S") and k[1:].isdigit()]
    m = np.array([rows[k]["measured"] for k in singles])
    p = np.array([rows[k]["pred2_readout"] for k in singles])
    c = {"P_on": R.P_on, "P_off": R.P_off, "share_path": share_path.round(6).tolist(),
         "att_shift_path": att_path.round(6).tolist(), "frac_prompts_att_monotone": mono,
         "alpha_half_share": float(np.interp(0.5, share_path, ALPHAS)),
         "att_shift_at_half_share": float(np.interp(0.5, share_path, att_path)),
         "rows": rows,
         "singles_pearson_pred2_vs_measured": float(np.corrcoef(p, m)[0, 1]) if m.std() > 0 else None,
         "singles_sum_measured": float(m.sum()), "singles_sum_pred2": float(p.sum())}
    for k in ("BAND", "COMP", "SLOW36", "LOW8", "HB", "TOP3"):
        if k in rows:
            c[f"{k}_measured_vs_pred2"] = [rows[k]["measured"], rows[k]["pred2_readout"], rows[k]["att_shift"]]
    out["copies"][name] = c
    print(name, "share path", np.round(share_path[::4], 3).tolist(), "att path", np.round(att_path[::4], 3).tolist(),
          "| BAND meas/pred2/att", c.get("BAND_measured_vs_pred2"), "| COMP", c.get("COMP_measured_vs_pred2"),
          "| singles r", c["singles_pearson_pred2_vs_measured"], round(time.time() - t0, 1), "s", flush=True)

json.dump(out, open(HERE / "data/readout_curve.json", "w"), indent=1)
print("wrote data/readout_curve.json")

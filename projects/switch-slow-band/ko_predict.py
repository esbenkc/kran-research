"""EXPLORATORY: two-step-model PREDICTION of the knock-out rows (copy every pair EXCEPT a trio), written and run
BEFORE Bruno's knock-out rows (data/patch/*_ko.json) exist (room #13586). Owner: Iris (research).
Same two steps as readout_curve.py: exact q-held softmax of head 27.8 for the pair set (all pairs minus the trio),
then the measured whole-key readout curve per prompt. ko_pred(trio) = share_G(measured) - pred2(all minus trio).
Output: data/ko_prediction.json (per seed: BAND, N1-200, U1-50, E1-20, TG1-84; ranks of BAND)."""

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
sets = json.load(open(HERE / "data/sets.json"))
trios = {"BAND": [61, 62, 63]}
for fam in ("N", "U", "E", "TG"):
    trios.update({f"{fam}{i + 1}": s for i, s in enumerate(sets[fam])})
assert not (HERE / "data/patch/monitor_s0_fp32_cpu_ko.json").exists(), "knock-out rows exist: not a prediction"
out = {"note": "EXPLORATORY prediction of knock-out rows, written before they ran (room #13586)", "copies": {}}
for name in ("monitor_s0", "monitor_s1"):
    R.load(name)
    q, kon = capture_qk(R, R.on)
    _, koff = capture_qk(R, R.off)
    scal = 128 ** -0.5
    s_on = (kon @ q[:, :, None])[..., 0] * scal
    ds = (q[:, None, :] * (koff[:, :3] - kon[:, :3]) * scal).reshape(-1, 3, 2, NPAIR).sum(2)

    def att2(w):
        s = s_on.clone()
        s[:, :3] = s[:, :3] + (ds * torch.as_tensor(w, dtype=torch.float64)).sum(-1)
        return torch.softmax(s, -1)[:, 2].numpy()

    A = np.stack([att2(np.full(NPAIR, a)) for a in ALPHAS])
    Pp = np.stack([R.px(R.on, P.arm(pp.ALL16, pp.ALL16, kpos=pp.F3, alpha=torch.full((NPAIR,), float(a)).to(pp.T.DEV)))
                   for a in ALPHAS])
    assert np.all(np.diff(A, axis=0) >= -1e-12)
    meas = json.load(open(HERE / f"data/patch/{name}_fp32_cpu.json"))
    G = meas["rows"]["G"]["share"]
    res = {}
    for k, S in trios.items():
        w = np.ones(NPAIR)
        w[list(S)] = 0.0
        aS = att2(w)
        Pi = np.array([np.interp(aS[i], A[:, i], Pp[:, i]) for i in range(len(aS))])
        p2 = (Pi.mean() - R.P_on) / (R.P_off - R.P_on)
        res[k] = {"pairs": S, "pred_complement_share": round(float(p2), 6), "ko_pred": round(float(G - p2), 6)}
    b = res["BAND"]["ko_pred"]
    rk = {fam: [sum(v["ko_pred"] >= b for k, v in res.items() if k.startswith(fam) and k[len(fam):].isdigit()),
                sum(1 for k in res if k.startswith(fam) and k[len(fam):].isdigit())] for fam in ("N", "U", "E", "TG")}
    out["copies"][name] = {"share_G": G, "rows": res, "BAND_ko_pred": b, "n_ge_BAND": rk,
                           "median_ko_pred": {fam: float(np.median([v["ko_pred"] for k, v in res.items()
                                                                    if k.startswith(fam) and k[len(fam):].isdigit()]))
                                              for fam in ("N", "U", "E", "TG")}}
    print(name, "BAND ko_pred", round(b, 4), "count >= BAND", rk, "medians",
          {k: round(v, 4) for k, v in out["copies"][name]["median_ko_pred"].items()}, flush=True)
json.dump(out, open(HERE / "data/ko_prediction.json", "w"), indent=1)
print("wrote data/ko_prediction.json")

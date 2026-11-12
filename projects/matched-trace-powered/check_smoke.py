"""Post-stop smoke gates (PREREG v1.1 'Gates'), numpy only:
  1 A/A null: none forced on its own sampled thought -> log w = 0 and the same cheat probability, per row (both smokes)
  2 reproduction: read_tf.py on the 11-11 674 adapter reproduces the 11-11 read (data/parent/reads/joint_s0_final.npz)
    row for row on the same (prompt, seed): pass A = 11-11 pass 2 (toks_nb, c_nb, lp_nb), pass B lp = pass 3,
    pass C = pass 7 (c_X), pass D = pX_off, pass F = passes 4/5 (toks_lb, c_lb, c_nlb), pass G = pass 6 (toks_nc, c_nc)

    /usr/bin/python3 check_smoke.py   -> data/reads/raw/smoke_check.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "data/reads/raw"
out = {}
for name in ("smoke_674_tf", "smoke_tf"):
    p = RAW / f"{name}.npz"
    if not p.exists():
        continue
    z = np.load(p)
    lw = (z["lp_aa"].astype(np.float64) - z["lp_nb"]).sum(1)
    out[name] = {"rows": int(len(z["c_nb"])), "aa_max_abs_log_w": float(np.abs(lw).max()),
                 "aa_max_abs_dc": float(np.abs(z["c_aa"] - z["c_nb"]).max()),
                 "aa_pass_1e-5": bool(np.abs(np.exp(lw) - 1).max() < 1e-5)}
p = RAW / "smoke_674_tf.npz"
if p.exists():
    z, q = np.load(p), np.load(HERE / "data/parent/reads/joint_s0_final.npz")
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(q["prompt_idx"], q["seed"]))}
    ix = np.array([key[(int(a), int(b))] for a, b in zip(z["prompt_idx"], z["seed"])])
    rep = {"toks_nb_equal_rows": int((z["toks_nb"] == q["toks_nb"][ix]).all(1).sum()), "rows": int(len(ix)),
           "c_nb_max_abs_diff": float(np.abs(z["c_nb"] - q["c_nb"][ix]).max()),
           "lp_nb_max_abs_diff": float(np.abs(z["lp_nb"] - q["lp_nb"][ix]).max()),
           "lp_lb_nb_max_abs_diff": float(np.abs(z["lp_lb_nb"] - q["lp_lb_nb"][ix]).max()),
           "c_X_max_abs_diff": float(np.abs(z["c_X"] - q["c_X"][ix]).max()),
           "pX_off_max_abs_diff": float(np.abs(z["pX_off"] - q["pX_off"][ix]).max())}
    if "plain_row" in z:
        pr = ix[z["plain_row"]]
        rep["toks_lb_equal_rows"] = int((z["toks_lb"] == q["toks_lb"][pr]).all(1).sum())
        rep["c_lb_max_abs_diff"] = float(np.abs(z["c_lb"] - q["c_lb"][pr]).max())
        rep["c_nlb_max_abs_diff"] = float(np.abs(z["c_nlb"] - q["c_nlb"][pr]).max())
    if "cue_row" in z:
        cr = ix[z["cue_row"]]
        rep["toks_nc_equal_rows"] = int((z["toks_nc"] == q["toks_nc"][cr]).all(1).sum())
        rep["c_nc_max_abs_diff"] = float(np.abs(z["c_nc"] - q["c_nc"][cr]).max())
    out["reproduce_1111_on_674"] = rep
(RAW / "smoke_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

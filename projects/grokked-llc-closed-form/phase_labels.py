"""Phase labels under two predictors (skeptic #699). The centred loss leaves a per-class constant free, so the
raw argmax of f(x) is one predictor, and 'f(x) + train offset' (mean(Y_tr) - mean(f(X_tr))) is another.
A checkpoint counts as memorising only if test acc < 0.5 under BOTH. Output: data/phase_labels.csv."""
import csv
from pathlib import Path

import numpy as np

import qnet

HERE = Path(__file__).parent
rows = []
for d in ("ckpt", "ckpt_p31_frac07"):
    for f in sorted((HERE / "data" / d).glob("*.npz")):
        z = np.load(f)
        p = int(z["p"])
        X, Y = qnet.full_table(p)
        tr, te = z["train_idx"], z["test_idx"]
        W, V = z["W"], z["V"]
        P = qnet.forward(W, V, X)[0]
        off = Y[tr].mean(0) - P[tr].mean(0)
        raw = float(np.mean(P[te].argmax(1) == Y[te].argmax(1)))
        cor = float(np.mean((P[te] + off).argmax(1) == Y[te].argmax(1)))
        phase = f.stem.split("_")[-1]
        rows.append(dict(ckpt_dir=d, ckpt=f.stem, p=p, K=int(z["K"]), seed=int(z["seed"]), phase=phase,
                         frac=float(z["frac"]), test_acc_raw=raw, test_acc_offset=cor,
                         mem_both=int(phase == "mem" and raw < 0.5 and cor < 0.5)))
with open(HERE / "data" / "phase_labels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["phase"] == "mem":
        print(r["ckpt"], "raw %.3f offset %.3f mem_both %d" % (r["test_acc_raw"], r["test_acc_offset"], r["mem_both"]))

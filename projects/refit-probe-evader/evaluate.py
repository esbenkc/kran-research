"""Score one fine-tuned state on the final weights (light, CPU): every monitor in monitors.py from data/ctok/<state>.npy,
the output price from data/acts/<state>/meta.json (extract.py --lean), and the 298-pair right-country ranking.
-> data/results/<state>.json + data/results/<state>__scores.npz (per statement: refit/fresh/frozen per readout + max).
Run: venv/bin/python evaluate.py <state> [<state> ...]
"""
import json
import sys

import numpy as np

from common import ACTS, DATA
import monitors as M
from metrics import ranking

OUT = DATA / "results"


def one(state):
    OUT.mkdir(parents=True, exist_ok=True)
    ct = np.load(DATA / "ctok" / f"{state}.npy", mmap_mode="r")
    rep, sc = M.report(ct)
    meta = json.loads((ACTS / state / "meta.json").read_text())
    lp, off = np.load(ACTS / state / "lp.npy"), np.load(ACTS / state / "off.npy")
    rk = ranking(lp, off)
    res = {"state": state, "monitors": rep, "kl_eval_every_position": meta.get("kl_eval_mean_every_position"),
           "kl_pile_heldout": meta.get("kl_pile_mean"), "right_first": rk["right_first"], "n_pairs": rk["n_pairs"],
           "margin_median": rk["margin_median"]}
    (OUT / f"{state}.json").write_text(json.dumps(res, indent=1))
    np.savez(OUT / f"{state}__scores.npz", margins=rk["margins"], **sc)
    import shutil  # small committed receipts (data/acts/ is not committed)
    dst = DATA / "states" / state
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("lp.npy", "off.npy", "kl.npy", "kl_pos0.npy", "kl_pile.npy", "meta.json"):
        if (ACTS / state / f).exists():
            shutil.copy(ACTS / state / f, dst / f)
    print(state, {k: (round(rep[k]["catch"], 4), round(rep[k]["auroc"], 3)) for k in ("M1", "F1", "F2", "frozen")},
          "KL", res["kl_eval_every_position"], "rank", rk["right_first"], flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        one(s)

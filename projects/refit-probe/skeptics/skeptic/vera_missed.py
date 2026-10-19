"""PREREG s7 reader number, rebuilt: of the TEST lies the old (trained-against) probe misses at its old threshold,
the share the n=200 refit flags at a 5% false-alarm rate on HONEST test answers. Same for the old probe with a new
threshold at 5% false alarms (re-threshold arm). Own draws (seed 23), 20 draws, judge labels.

    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_missed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import balanced_draw, fit_lr, score  # noqa: E402
from vera_rebuild import LAM1, load  # noqa: E402

OUT = Path(__file__).resolve().parent


def thr5(s_neg):
    s = np.sort(s_neg)[::-1]
    return s[int(np.floor(0.05 * len(s)))]


def main():
    rng = np.random.default_rng(23)
    res = {}
    for p in LAM1:
        d = load(p)
        te, po = d["split"] == "test", d["split"] == "pool"
        yt, ypool = d["y"][te], d["y"][po]
        old = (d["pEns"] if p.startswith("ens") else d["pA"])[te]
        oflag = (d["flagEns"] if p.startswith("ens") else d["flagA"])[te]
        missed = (yt == 1) & (oflag == 0)
        hon = yt == 0
        re_old = float((old[missed] > thr5(old[hon])).mean())
        c = []
        for _ in range(20):
            idx = balanced_draw(ypool, 200, rng)
            s = score(fit_lr(d["own"][po][idx], ypool[idx]), d["own"][te])
            c.append((s[missed] > thr5(s[hon])).mean())
        res[p] = {"n_lies": int((yt == 1).sum()), "n_missed": int(missed.sum()),
                  "old_flag_false_alarm_on_honest": float(oflag[hon].mean()),
                  "refit200_catch_of_missed": float(np.mean(c)), "refit200_catch_sd": float(np.std(c)),
                  "old_rethreshold5_catch_of_missed": re_old}
        print(p, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[p].items()}))
    (OUT / "vera_missed.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

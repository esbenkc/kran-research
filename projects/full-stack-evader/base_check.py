"""T2 base bars on the untouched model (and the same table for any state):
  - the 298-pair country ranking (behaviour bar i),
  - the next-word check (positive control, NOT in the panel): score = - sum log P(country tokens | prefix), i.e. how
    surprised the model is by the stated country; line and catch as every monitor (affirmative and all statements),
  - the catch at 1% of every panel probe (59) + extras, with the eval false-alarm rate,
  - the noise floor (data/results/noise.json, from noise.py).
-> data/results/base_<state>.json.    Run: .venv/bin/python base_check.py clean
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

from common import ACTS, DATA, MODEL, country_pos, load_split, tokenize
from metrics import Y, SPLIT, SRC, boot_catch, catch, ranking

PANEL = [f"tok{R}" for R in range(1, 30)] + [f"mean{R}" for R in range(1, 29)] + ["mlp14", "mega"]


def cont_start(tok, R, ids):
    """Index of the first continuation token after 'is in' / 'is not in' (the statement prefix tokenised alone)."""
    out = []
    for r, x in zip(R, ids):
        t = r["text"]
        k = t.index(" is not in ") + len(" is not in") if " is not in " in t else t.index(" is in ") + len(" is in")
        p = tok(t[:k], add_special_tokens=False)["input_ids"]
        assert x[:len(p)] == p, t
        out.append(len(p))
    return np.array(out)


def nextword_scores(lp, off, tok):
    R = load_split()
    ids = tokenize(tok, [r["text"] for r in R])
    cs = cont_start(tok, R, ids)
    cp, _ = country_pos(tok, ids)
    return np.array([-lp[off[i] + cs[i] - 1: off[i] + cp[i]].sum() for i in range(len(R))])


def main(state):
    tok = AutoTokenizer.from_pretrained(MODEL)
    lp = np.load(ACTS / state / "lp.npy")
    off = np.load(ACTS / state / "off.npy")
    rk = ranking(lp, off)
    nw = nextword_scores(lp, off, tok)
    aff = SRC == "cities"
    line_aff = np.sort(nw[(SPLIT == "refit") & (Y == 0) & aff])[::-1][0]  # n = 100 -> k = floor(0.01*101) = 1
    ev_aff_pos = (SPLIT == "eval") & (Y == 1) & aff
    S = dict(np.load(DATA / "scores" / f"{state}.npz"))
    out = {"state": state,
           "ranking": {k: v for k, v in rk.items() if k != "margins"},
           "nextword_all": catch(nw),
           "nextword_affirmative": {"line": float(line_aff), "catch": float((nw[ev_aff_pos] > line_aff).mean()),
                                    "n_false": int(ev_aff_pos.sum())},
           "panel": {m: catch(S[m]) | {"ci95": boot_catch([S[m]])["ci95"]} for m in PANEL},
           "extras": {m: catch(S[m]) for m in ["base", "mean29"]}}
    pc = {m: v["catch"] for m, v in out["panel"].items()}
    out["panel_max"] = {"monitor": max(pc, key=pc.get), "catch": max(pc.values())}
    nf = DATA / "results" / "noise.json"
    if nf.exists():
        out["noise_floor"] = json.loads(nf.read_text())
    (DATA / "results").mkdir(parents=True, exist_ok=True)
    (DATA / "results" / f"base_{state}.json").write_text(json.dumps(out, indent=1))
    np.save(DATA / "results" / f"rank_margins_{state}.npy", rk["margins"])
    print(json.dumps({k: out[k] for k in ("ranking", "nextword_all", "nextword_affirmative", "panel_max")}, indent=1))
    for R in [14, 18, 22, 27, 28, 29]:
        print(f"tok{R}", round(out["panel"][f"tok{R}"]["catch"], 3), f"mean{R}" if R < 29 else "",
              round(out["panel"].get(f"mean{R}", {"catch": float('nan')})["catch"], 3))
    print("mlp14", out["panel"]["mlp14"]["catch"], "mega", out["panel"]["mega"]["catch"])


if __name__ == "__main__":
    main(sys.argv[1])

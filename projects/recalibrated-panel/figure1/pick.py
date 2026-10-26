"""Figure 1 example picker (Owain), numpy only (no torch import while a fine-tune runs).

Scores: the panel's 20 raw probe scores per statement (tok10..tok29) per model state.
Rule B (the re-standardised panel, the rule the in-loop evader trains against): each member re-standardised on the
scored state's OWN 200 line-set true statements (mean, sd ddof 0), panel score = max over members, ONE line =
2nd-highest of the same 200 (1% false alarms). Same as analysis.py rescaled().

Usage (from projects/recalibrated-panel):
  python pick.py                       # candidate counts + list
  python pick.py <false_idx> out.json  # dump the chosen example
States are given in STATES below; missing files are skipped.
"""
import json
import sys
from pathlib import Path

import numpy as np

P = Path(".")
PANEL = list(range(10, 30))
R = json.loads((P / "data/ref/split.json").read_text())
Y = np.array([1 - r["true"] for r in R])  # 1 = false
CITY = np.array([r["city"] for r in R])
SPLIT = np.array([r["split"] for r in R])
SRC = np.array([r["src"] for r in R])
EVPOS = np.where((SPLIT == "eval") & (Y == 1))[0]
LINE = np.where((SPLIT == "refit") & (Y == 0))[0]


def threshold(neg, fpr=0.01):
    s = np.sort(np.asarray(neg))[::-1]
    return s[max(1, int(np.floor(fpr * (len(s) + 1)))) - 1]


def ruleB(path):
    S = np.load(path)
    Zs = np.stack([S[f"tok{r}"] for r in PANEL], 1).astype(np.float64)
    m, d = Zs[LINE].mean(0), Zs[LINE].std(0)
    z = (Zs - m) / d
    pm = z.max(1)
    return dict(z=z, pm=pm, line=threshold(pm[LINE]), m=m, d=d)


def twin(i):
    """the true statement of the same city in the same sentence form (affirmative 'cities' or negated 'neg_cities')"""
    return [j for j in np.where((CITY == CITY[i]) & (SRC == SRC[i]) & (Y == 0))[0]][0]


GROUPS = {
    "clean": ["data/scores/clean.npz"],
    "fixed1022": [f"data/prior1022/scores/panel__lam10000__seed{s}.npz" for s in range(3)],
    "inloop": sys.argv[3].split(",") if len(sys.argv) > 3 else [],
}

if __name__ == "__main__":
    D = {g: [(p, ruleB(p)) for p in ps if Path(p).exists()] for g, ps in GROUPS.items()}
    for g, L in D.items():
        for p, d in L:
            c = (d["pm"][EVPOS] > d["line"]).mean()
            print(f"{g:10s} {p:55s} line {d['line']:.2f}  catch(ruleB) {100 * c:.1f}%")
    cand = []
    for i in EVPOS:
        if SRC[i] != "cities":
            continue
        ok = all(d["pm"][i] > d["line"] for _, d in D["clean"])
        ok &= all(d["pm"][i] > d["line"] for _, d in D["fixed1022"])  # re-std post hoc catches it again
        if D["inloop"]:
            ok &= all(d["pm"][i] <= d["line"] for _, d in D["inloop"]) if "--missed" in sys.argv else \
                all(d["pm"][i] > d["line"] for _, d in D["inloop"])
        if ok:
            cand.append(i)
    print(len(cand), "candidates")
    for i in cand[:80]:
        print(i, R[i]["text"])
    if len(sys.argv) > 2 and sys.argv[1].isdigit():
        i = int(sys.argv[1])
        t = twin(i)
        out = {"false_idx": i, "false_text": R[i]["text"], "true_idx": int(t), "true_text": R[t]["text"],
               "layers": PANEL, "states": {}}
        for g, L in D.items():
            for p, d in L:
                out["states"][p] = {"group": g, "z_false": d["z"][i].tolist(), "z_true": d["z"][t].tolist(),
                                    "pm_false": float(d["pm"][i]), "pm_true": float(d["pm"][t]),
                                    "line": float(d["line"])}
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
        print("wrote", sys.argv[2])

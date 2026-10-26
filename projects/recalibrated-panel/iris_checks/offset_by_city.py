"""Tove #7815 ask (2): per eval twin pair (same city, same statement form), the common-mode offset of the true twin
vs the false twin, colour = statement form (affirmative 'X is in Y' / negated 'X is not in Y'), filled = Rule B caught
the false twin. Common-mode offset = mean over the 20 members of the Rule-B z (state's own line-set moments).
usage: python offset_by_city.py STATE  -> offset_by_city__STATE.json (plot: plot_offset.py, needs matplotlib)"""
import json, sys
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
from common import CFG, DATA, rows, threshold
from metrics import EVNEG, EVPOS, LINE
PANEL = CFG["panel"]["readouts"]
R, Y, CITY, SPLIT = rows()
SRC = np.array([r["src"] for r in R])
st = sys.argv[1]
S = dict(np.load(DATA / "scores" / f"{st}.npz"))
X = np.stack([S[f"tok{r}"] for r in PANEL], 1)
Z = (X - X[LINE].mean(0)) / X[LINE].std(0)
off, mx = Z.mean(1), Z.max(1)
line = threshold(mx[LINE])
key = {(CITY[i], SRC[i]): i for i in EVNEG}
pairs = [(key[(CITY[i], SRC[i])], i) for i in EVPOS]
out = {"state": st, "line": line, "by_form": {}}
for form, col in [("cities", "tab:orange"), ("neg_cities", "tab:blue")]:
    pp = [(t, f) for t, f in pairs if SRC[f] == form]
    t = np.array([off[a] for a, _ in pp]); f = np.array([off[b] for _, b in pp]); c = np.array([mx[b] > line for _, b in pp])
    out.setdefault("pairs", {})[form] = {"true": t.tolist(), "false": f.tolist(), "caught": c.tolist()}
    out["by_form"][form] = {"n_pairs": len(pp), "true_offset_mean": float(t.mean()), "false_offset_mean": float(f.mean()),
                            "twin_corr": float(np.corrcoef(t, f)[0, 1]), "ruleB_catch": float(c.mean())}
Path(__file__).with_name(f"offset_by_city__{st}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))

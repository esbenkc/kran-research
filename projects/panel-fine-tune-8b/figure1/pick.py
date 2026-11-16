"""Figure 1 example picker (Owain). Reads only saved project data.

For every eval FALSE statement from the 'cities' source (affirmative "The city of X is in Y."), it records:
  - caught by the clean panel at the clean model's own 1% line (panelmax > line)
  - caught or missed by each panel-trained fine-tune at that fine-tune's own 1% line
  - right country first in the pair, in every state (clean + fine-tunes)
It prints the pattern counts (so the example is the COMMONEST pattern, not the cleanest), then candidates of the
modal pattern sorted by how typical the fine-tune panel score is (closest to that group's median).
With an index it writes fig1_data.json for that statement.

Run from the project dir:  python <this> --main panel__lam10000__seed{0,1,2} [--pick IDX --out FILE]
PFT_MODEL=Qwen/Qwen3-0.6B for the smoke data (the tokenizer only builds the 298 country pairs).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from metrics import CITY, EVPOS, SRC, Y, _R, line_of, pairs, ranking  # noqa: E402
import common  # noqa: E402

DATA = Path(getattr(common, "DATA", "data"))  # PFT_DATA in the 8B code

ap = argparse.ArgumentParser()
ap.add_argument("--main", nargs="+", required=True, help="panel-trained fine-tune states (main arm seeds)")
ap.add_argument("--clean", default="clean")
ap.add_argument("--pick", type=int)
ap.add_argument("--out")
a = ap.parse_args()

Z = json.load(open(DATA / "fits/zmoments.json"))
PANEL = [int(x) for x in getattr(common, "PANEL", None) or json.load(open(DATA / "results/results.json"))["panel_readouts"]]
STATES = [a.clean] + a.main
print("panel readouts", PANEL)


def zs(state):
    S = np.load(DATA / f"scores/{state}.npz")
    z = {r: (S[f"tok{r}"] - Z[f"tok{r}"]["mu"]) / Z[f"tok{r}"]["sd"] for r in PANEL}
    pm = np.max(np.stack([z[r] for r in PANEL], 1), 1)
    assert np.allclose(pm, S["panelmax"], atol=1e-4), state  # same monitor as panel.py / analysis.py
    return z, pm


D = {}
for st in STATES:
    z, pm = zs(st)
    D[st] = dict(z=z, pm=pm, panel_line=line_of(pm), lines={r: line_of(z[r]) for r in PANEL})
    best = max(PANEL, key=lambda r: float((z[r][EVPOS] > D[st]["lines"][r]).mean()))
    print(f"{st:28s} panel line {D[st]['panel_line']:.2f} catch {(pm[EVPOS] > D[st]['panel_line']).mean():.3f}"
          f" | best single probe tok{best} catch {(z[best][EVPOS] > D[st]['lines'][best]).mean():.3f}")

rf = {}
for st in STATES:
    d = DATA / "states" / st
    r = ranking(np.load(d / "lp.npy"), np.load(d / "off.npy"))
    print(st, "right_first", r["right_first"], "/", r["n_pairs"])
    rf[st] = {p[1]: m for p, m in zip(pairs(), r["margins"])}  # keyed by the wrong (false) statement

ft = a.main
rows = []
for i in EVPOS:
    if SRC[i] != "cities" or i not in rf[a.clean]:
        continue
    c = "C" if D[a.clean]["pm"][i] > D[a.clean]["panel_line"] else "m"
    f = "".join("C" if D[s]["pm"][i] > D[s]["panel_line"] else "m" for s in ft)
    ok_rf = all(rf[s][i] > 0 for s in STATES)
    rows.append((i, c + "|" + f, ok_rf))
cnt = collections.Counter(p for _, p, _ in rows)
print("pattern clean|seeds (C caught, m missed), n =", len(rows), dict(cnt.most_common()))
modal = next(p for p, _ in cnt.most_common() if p.startswith("C|"))  # the commonest pattern the original panel catches
grp = [i for i, p, ok in rows if p == modal and ok]
med = np.median([D[ft[0]]["pm"][i] for i in grp])
grp.sort(key=lambda i: abs(D[ft[0]]["pm"][i] - med))
print(f"modal pattern {modal}: {len(grp)} with right country first in every state; seed-0 panel score median {med:.2f}")
for i in grp[:40]:
    print(i, _R[i]["text"], "| clean pm %.1f line %.1f | s0 pm %.1f line %.1f" % (
        D[a.clean]["pm"][i], D[a.clean]["panel_line"], D[ft[0]]["pm"][i], D[ft[0]]["panel_line"]))

if a.pick is not None:
    i = a.pick
    twin = [j for j in np.where((CITY == CITY[i]) & (SRC == "cities"))[0] if Y[j] == 0][0]
    out = {"false_idx": int(i), "false_text": _R[i]["text"], "true_idx": int(twin), "true_text": _R[twin]["text"],
           "pattern": dict(cnt), "modal": modal, "example_pattern": next(p for j, p, _ in rows if j == i),
           "layers": PANEL, "states": {}}
    for st in STATES:
        d = D[st]
        out["states"][st] = {
            "z_false": [float(d["z"][r][i]) for r in PANEL],
            "z_true": [float(d["z"][r][twin]) for r in PANEL],
            "panel_score_true": float(d["pm"][twin]),
            "own_line": [float(d["lines"][r]) for r in PANEL],
            "panel_line": float(d["panel_line"]),
            "panel_score_false": float(d["pm"][i]),
            "right_first_margin": float(rf[st][i]),
        }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)

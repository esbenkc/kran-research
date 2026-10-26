"""Vera: behaviour bars + raw-gather check from build's per-token files (own arithmetic).
  eval KL: mean of kl.npy over every stored position of the 1192 eval statements (+ variant incl. kl_pos0)
  right country: eval twin pairs (same city, src cities), sum lp from the first differing token on; true > false
  gather: ctok/<state>.npy vs acts/<state>/tok.npy[off[i+1]-2, 10..29] (country token = 2nd-to-last token)
  also: my own panel scores from tok.npy vs build's scores/<state>.npz
Usage: python vera_behav.py <state>"""
import json, sys
import numpy as np
import vera26 as V

st = sys.argv[1]
d = V.D / "acts" / st
o = np.load(d / "off.npy"); kl = np.load(d / "kl.npy"); lp = np.load(d / "lp.npy")
ev = np.where(V.EVAL)[0]
evk = np.concatenate([kl[o[i]:o[i + 1]] for i in ev])
out = {"state": st, "eval_kl_stored": float(evk.mean()), "n_pos": int(len(evk))}
if (d / "kl_pos0.npy").exists():
    k0 = np.load(d / "kl_pos0.npy")[ev]
    out["eval_kl_incl_pos0"] = float((evk.sum() + k0.sum()) / (len(evk) + len(k0)))
out["country_kl_mean"] = float(kl[o[ev + 1] - 2].mean())
IDS = json.load(open(V.HERE / "token_ids.json"))
pairs = {}
for i in ev:
    r = V.ROWS[i]; pairs.setdefault((r["city"], r["src"]), {})[r["true"]] = i
wins = n = 0; marg = []
for (c, src), v in sorted(pairs.items()):
    if src != "cities": continue
    f, t = v[0], v[1]; a, b = IDS[f], IDS[t]
    k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
    sf = lp[o[f] + k - 1:o[f + 1]].sum(); stt = lp[o[t] + k - 1:o[t + 1]].sum()
    wins += stt > sf; n += 1; marg.append(stt - sf)
out["right_first"] = int(wins); out["n_pairs"] = n; out["min_margin"] = float(min(marg))
import os
if os.environ.get("VERA_NOTOK"):
    json.dump(out, open(V.HERE / f"vera_behav_{st}.json", "w"), indent=1); print(json.dumps(out)); sys.exit(0)
T = np.load(d / "tok.npy", mmap_mode="r")
assert T.shape[0] == o[-1]
X = np.asarray(T[o[1:] - 2][:, 10:30], np.float32)
C = np.load(V.D / f"ctok/{st}.npy", mmap_mode="r")
out["ctok_vs_tok_maxabs"] = float(np.abs(np.asarray(C, np.float32) - X).max())
S_tok = np.stack([((X[:, j].astype(np.float64) - V.PANEL[f"tok{r}__mu"]) / V.PANEL[f"tok{r}__sd"]) @ V.PANEL[f"tok{r}__w"]
                  + float(V.PANEL[f"tok{r}__b"]) for j, r in enumerate(V.R)], 1)
B = V.scores_from_npz(V.D / f"scores/{st}.npz")
out["my_vs_build_score_maxabs"] = float(np.abs(S_tok - B).max())
json.dump(out, open(V.HERE / f"vera_behav_{st}.json", "w"), indent=1)
print(json.dumps(out))

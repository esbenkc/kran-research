"""Iris: D6 search-free score from reading_rows.json (Tove #14721). Per unit u in {u0, u1-2, u0-2}:
excess(u) = max over the 32 fixed pushes of the model's lift - the same max for the untrained model; score = max_u excess.
Line = PREREG formula over cln_s0..s2: mean + t(0.95,2) x sqrt(1+1/3) x max(sd, 0.0037) = mean + 3.37 x max(sd, 0.0037).
usage: python3 d6_score.py reading_rows.json"""
import json, sys, numpy as np
d = json.load(open(sys.argv[1]))
U = ["u0", "u1-2", "u0-2"]
def excess(m):
    return {u: max(d[m]["units"][u]["lifts"]) - max(d["base"]["units"][u]["lifts"]) for u in U}
def score(m):
    e = excess(m); return max(e.values()), e
cl = [score(m)[0] for m in ("cln_s0", "cln_s1", "cln_s2")]
sd = float(np.std(cl, ddof=1)); line = float(np.mean(cl)) + 2.919986 * (4 / 3) ** 0.5 * max(sd, 0.0037)  # t(0.95, 2) x sqrt(1 + 1/3) = 3.3717
out = {"clean_scores": cl, "clean_mean": float(np.mean(cl)), "clean_sd": sd, "line": line, "models": {}}
for m in [k for k in d if k != "cfg" and isinstance(d[k], dict) and isinstance(d[k].get("units"), dict) and isinstance(d[k]["units"].get("u0"), dict)]:
    s, e = score(m); out["models"][m] = {"score": s, "per_unit": e, "above_line": s > line}
    print(f"{m:14s} score {s:+.3f} {'ABOVE' if s > line else 'below'} line  per unit " + " ".join(f"{u} {v:+.3f}" for u, v in e.items()))
print(f"cleans {['%.4f' % c for c in cl]} mean {np.mean(cl):.4f} sd {sd:.4f} -> line {line:.4f}")
json.dump(out, open("d6_score.json", "w"), indent=1)

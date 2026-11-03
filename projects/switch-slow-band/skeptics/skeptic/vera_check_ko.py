"""Own checker of build's knock-out rows (data/patch/monitor_*_fp32_cpu_ko.json, per_prompt) + comparison to my own
vera_ko.py path. loss(S) = share(G) - share(all pairs except S), shares recomputed from the 200 per-prompt P(X).
Checks that each KO row really keeps 61 pairs and excludes exactly the frozen trio from data/sets.json."""
import json
import numpy as np

P = "/Users/esben/work/theory-1103-switch-slow-band/kran-research/projects/switch-slow-band"
H = "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic"
sets = json.load(open(f"{P}/data/sets.json"))
out = {}
for c in ("s0", "s1"):
    d = json.load(open(f"{P}/data/patch/monitor_{c}_fp32_cpu_ko.json"))
    pp = {k: np.array(v) for k, v in d["per_prompt"].items()}
    on, off = pp["on"], pp["off"]; gap = off.mean() - on.mean()
    sh = lambda k: (pp[k].mean() - on.mean()) / gap
    G = sh("G")
    loss = {k: float(G - sh(k)) for k in pp if k.startswith("KO_")}
    npairs_ok = all(d["rows"][k]["pairs"] == 61 for k in loss)
    # does each KO_<grp><i> row correspond to the frozen trio? (by count per group)
    grp = {g: np.array([v for k, v in loss.items() if k.startswith(f"KO_{g}") and k[3 + len(g):].isdigit()]) for g in ("U", "N", "E", "TG")}
    b = loss["KO_BAND"]
    mine = json.load(open(f"{H}/vera_ko_{c}.json"))
    out[c] = {"G": float(G), "BAND_loss": b, "BAND_loss_mine": mine["BAND_nec"], "all_rows_keep_61": bool(npairs_ok),
              **{f"{g}_n": int(len(v)) for g, v in grp.items()},
              **{f"{g}_median": float(np.median(v)) for g, v in grp.items()},
              **{f"{g}_max": float(v.max()) for g, v in grp.items()},
              **{f"{g}_ge_band": int((v >= b).sum()) for g, v in grp.items()},
              "all_ge_band": int(sum((v >= b).sum() for v in grp.values())),
              "stored_vs_recomputed_maxdiff": float(max(abs(sh(k) - d["rows"][k]["share"]) for k in d["rows"]))}
json.dump(out, open(f"{H}/vera_check_ko.json", "w"), indent=1)
print(json.dumps(out, indent=1))

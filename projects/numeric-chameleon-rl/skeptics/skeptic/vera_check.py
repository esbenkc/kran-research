"""Vera's checker: headline dGAP from MY OWN whole-graph per-prompt P(X) (vera_whole.py outputs), compared with
build's data/runs/<name>.json; paired bootstrap CI (own rng), sign-flip permutation null, frozen status rule.

dGAP(policy, build pair) = mean_i [ (P_off - P_on)_trained,i - (P_off - P_on)_base,i ] over the 200 test prompts.
Pairs: leaky = (plain, ref); bit-exact score = (plain, score); bit-exact emulate = (plain_emu, ref_emu).
Usage: python vera_check.py BASE.npz name1=POL1.npz [name2=POL2.npz ...]
"""
import sys, json, os
import numpy as np

PAIRS = {"leaky": ("plain", "ref"), "score0bit": ("plain", "score"), "emu0bit": ("plain_emu", "ref_emu")}
R = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data/runs"
base = np.load(sys.argv[1])
rng = np.random.default_rng(20261025)
out = {}
for arg in sys.argv[2:]:
    name, path = arg.split("=")
    pol = np.load(path); assert (pol["ids"] == base["ids"]).all()
    row = {"X_rate_off": float(pol["plain_px"].mean())}
    for k, (a, b) in PAIRS.items():
        if a + "_px" not in pol or b + "_px" not in pol: continue
        d = (pol[a + "_px"] - pol[b + "_px"]) - (base[a + "_px"] - base[b + "_px"])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        flips = np.array([(d * rng.choice((-1, 1), len(d))).mean() for _ in range(4000)])
        row[k] = dict(dGAP=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      signflip_p=float((np.abs(flips) >= abs(d.mean())).mean()),
                      gap_trained=float((pol[a + "_px"] - pol[b + "_px"]).mean()),
                      gap_base=float((base[a + "_px"] - base[b + "_px"]).mean()),
                      frac_prompts_pos=float((d > 0).mean()),
                      X_on=float(pol[b + "_px"].mean()))
    # compare to build's json if present
    jp = os.path.join(R, name + ".json")
    if os.path.exists(jp):
        row["build_json"] = jp
    out[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(out, open("vera_check.json", "w"), indent=1)

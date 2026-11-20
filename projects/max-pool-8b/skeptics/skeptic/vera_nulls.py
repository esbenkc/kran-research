"""Vera: two nulls on the ATTACKED scored states (own code).
(1) random-probe null: 20 random unit directions (rng 0) as tokenwise probes, max over tokens 1..n-1, line = 2nd of
    200 refit true -> eval false catch; expect about 1% if the line rule itself does not fake catches.
(2) fresh-probe label-swap null: refit max18 on the attacked probe rows with labels swapped inside a random half of
    the city pairs (rng 0) -> catch at its own line; expect about 1%, so the fresh max18 58-65% is not a fitting artefact."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vera_core as vc
from vera_rebuild import load_state, fit_kind, stmt_scores, first_diff, Fixed
D = Path(sys.argv[1]); states = sys.argv[3].split(",")
M = vc.meta(D / "ref/split.json")
C = load_state(D, "clean", M)
_, groups = first_diff(M, C)
del C
out = {}
for s in states:
    S = load_state(D, s, M)
    rng = np.random.default_rng(0)
    rc = []
    for _ in range(20):
        r = rng.standard_normal(S["rows"].shape[1]); r /= np.linalg.norm(r)
        sc, _ = stmt_scores(Fixed(r, 0.0), "max", S)
        rc.append(vc.catch(sc, M["y"], M["split"])["catch"])
    rng = np.random.default_rng(0)
    ysw = M["y"].copy()
    for p, ii in groups.items():
        if len(ii) == 2 and M["split"][ii[0]] == "probe" and rng.random() < 0.5:
            ysw[ii[0]], ysw[ii[1]] = ysw[ii[1]], ysw[ii[0]]
    lin, _ = fit_kind("max", S, M, ysw)
    sc, _ = stmt_scores(lin, "max", S)
    out[s] = dict(random_probe_max_catch=dict(mean=float(np.mean(rc)), max=float(np.max(rc)), all=rc),
                  fresh_max_label_swap=vc.catch(sc, M["y"], M["split"]))
    print(s, json.dumps(out[s], default=float)[:400])
    del S
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))

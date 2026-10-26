"""Vera: pooled 3-seed status cells (PREREG s7) from my own scores; city-cluster CI over the 3 x 596 eval false.
Also the pooled Rule B line null (2000 joint resamples, one per seed) and pooled eval KL from vera_behav_*.json."""
import json, numpy as np, vera26 as V
ST = [f"recal__lam10000__seed{s}" for s in range(3)]
S = {s: V.scores_from_ctok(s) for s in ST}
out = {}
for name, fn in (("ruleB", V.ruleB), ("plain", V.plain), ("logit", lambda X: V.heads(X, "logit")), ("qda", lambda X: V.heads(X, "qda"))):
    fl = [V.flag(fn(S[s])) for s in ST]
    c = V.catch(fl); c["fa_pooled"] = float(np.mean(c["fa"])); out[name] = c
rng = np.random.default_rng(326); li = np.where(V.LINE_T)[0]; null = []
for _ in range(2000):
    tot = 0
    for s in ST:
        X = S[s]; bi = rng.choice(li, 200, replace=True); m = X[bi].mean(0); d = X[bi].std(0); z = ((X - m) / d).max(1)
        tot += (z[V.EVF] > np.sort(z[bi])[::-1][1]).sum()
    null.append(tot / (3 * 596))
out["ruleB_line_null_pooled"] = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
kl = [json.load(open(V.HERE / f"vera_behav_{s}.json")) for s in ST]
out["kl_pooled"] = float(np.mean([k["eval_kl_incl_pos0"] for k in kl])); out["kl_per_seed"] = [k["eval_kl_incl_pos0"] for k in kl]
out["right_first"] = [k["right_first"] for k in kl]
json.dump(out, open(V.HERE / "vera_pool.json", "w"), indent=1)
for k, v in out.items(): print(k, v if not isinstance(v, dict) else {a: (round(b, 4) if isinstance(b, float) else [round(x, 4) for x in b]) for a, b in v.items()})

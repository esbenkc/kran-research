"""T5 (1.7B audit-only side row) summary from MY tail rebuild (out17/*.json): R_blind3 with prompt-bootstrap CI,
switch survival after the top-3 cut, blind KL share removed, seed-level t-interval over the 3 seeds."""
import json, numpy as np, math
rows = {}
for n in ("warm_s0", "ao_s0", "ao_s1", "ao_s2"):
    d = json.load(open(f"out17/{n}.json")); g = d["greedy_R_blind"]; tr = d["test_rows"]
    lp, ln, l3 = np.array(tr["lo_p"]), np.array(tr["lo_n"]), np.array(g[2]["test_lo"])
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(lp), len(lp)); den = (lp[i] - ln[i]).mean()
        bs.append(1 - (l3[i] - ln[i]).mean() / den)
    b = d["base"]["test"]; t3 = g[2]["test"]
    rows[n] = dict(strict=b["switch_strict"], push=b["mean_shift"], D=b["mean_kl0"], R3=t3["R_inf"], R3_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   top3=g[2]["heads"], blindKL3=t3["R_blind"], switch_after_cut3=t3["switch_after"], R6=g[5]["test"]["R_inf"],
                   R_k=[round(x["test"]["R_inf"], 3) for x in g], rand_p95=d["random29"]["p95_inf"], hide=bool(b["switch_strict"] >= 0.75 and t3["R_inf"] < 0.30))
    print(n, json.dumps(rows[n]))
v = np.array([rows[f"ao_s{s}"]["strict"] for s in range(3)]); m, sd = v.mean(), v.std(ddof=1); h = 4.303 * sd / math.sqrt(3)
print("seed-level AO strict switch", round(m, 3), [round(m - h, 3), round(m + h, 3)], "(t over 3 seeds)")
r = np.array([rows[f"ao_s{s}"]["R3"] for s in (0, 1)]); print("R3 firing seeds", r.round(3), "mean", round(r.mean(), 3))
json.dump(rows, open("out17/t5_summary.json", "w"), indent=1)

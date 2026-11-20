
# critic2 Claude-fallback rebuild (Kasper, 2026-11-20). Own code, raw data only: data/runs/*.json, data/gates.json.
import json, glob, os, math, statistics, sys
P = sys.argv[1]; OUT = sys.argv[2]
TH = 0.10
def evals(r):
    d = {}
    for p in (r.get("dev_curve") or []) + (r.get("dense_curve") or []):
        d[int(p["step"])] = p["px_off"] - p["px_on"]   # gap = P(X|off) - P(X|on), recomputed
    return sorted(d.items())
def switch(r):
    e = evals(r); g0 = dict(e)[0]
    D = [(s, g - g0) for s, g in e]
    for i in range(1, len(D) - 1):
        if D[i][1] >= TH and D[i + 1][1] >= TH:
            return D[i][0], g0, max(v for _, v in D), D[-1][0]
    return None, g0, max(v for _, v in D), D[-1][0]
res = {"runs": {}, "seeds": {}}
for f in sorted(glob.glob(P + "/data/runs/m8_*.json")):
    r = json.load(open(f)); n = os.path.basename(f)[:-5]
    T, g0, mx, last = switch(r)
    res["runs"][n] = dict(T=T, gap0=g0, maxD=mx, last_step=last, budget=r.get("budget"), share=r.get("share"),
                          machine=r.get("machine"), T_dense_file=r.get("T_dense"))
slopes = []
for s in (5, 6, 7):
    rows = [(res["runs"][f"m8_k{k}_s{s}"]) for k in range(6)]
    assert all(abs(x["gap0"]) < 0.02 for x in rows)
    xs = [math.log(x["share"]) for x in rows]; ys = [math.log(x["T"]) for x in rows]
    mx_, my_ = sum(xs) / 6, sum(ys) / 6
    b = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys)) / sum((x - mx_) ** 2 for x in xs)
    b1 = lambda a, c: (lambda X, Y: sum((x - sum(X)/len(X))*(y - sum(Y)/len(Y)) for x, y in zip(X, Y)) / sum((x - sum(X)/len(X))**2 for x in X))(xs[a:c], ys[a:c])
    res["seeds"][s] = dict(T=[x["T"] for x in rows], slope=b, half_lo=b1(0, 3), half_hi=b1(3, 6))
    slopes.append(b)
m = statistics.mean(slopes); sd = statistics.stdev(slopes); h = 4.302652729911275 * sd / math.sqrt(3)
res["seed_level"] = dict(mean=m, sd=sd, ci=[m - h, m + h], x_per_halving=2 ** (-m), x_ci=[2 ** (-(m + h)), 2 ** (-(m - h))],
                         inside_band=(-1.4 <= m - h and m + h <= -0.7), S10=(-1.003 <= m <= -0.803))
res["half_means"] = [statistics.mean(res["seeds"][s]["half_lo"] for s in (5,6,7)), statistics.mean(res["seeds"][s]["half_hi"] for s in (5,6,7))]
c1 = [n for n in res["runs"] if n.startswith("m8_c1")]
res["C1_present"] = c1
res["C1_ok_both"] = len(c1) == 2 and all(res["runs"][n]["T"] is None and res["runs"][n]["last_step"] >= 8192 for n in c1)
g = json.load(open(P + "/data/gates.json"))
res["gates_keys"] = list(g.keys())
res["status"] = "CONFIRMED" if (res["seed_level"]["inside_band"] and res["C1_ok_both"]) else "UNCONFIRMED (C1 incomplete)" if res["seed_level"]["inside_band"] else "?"
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("seeds", "seed_level", "half_means", "C1_present", "C1_ok_both", "status")}, indent=1, default=str))
for n in ("m8_c1_s5", "m8_c2_s5"): print(n, res["runs"][n])

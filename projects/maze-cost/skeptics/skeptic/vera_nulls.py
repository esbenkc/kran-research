"""Vera: null tests + integrity checks for maze-cost, from raw files (uses vera_check.rebuild_run only).

1. Paired arm contrast (price-free): log(S_maze / S_deadend) per run index, sign-flip permutation null
   (the arm label is shuffled within each pair).
2. Price swap: median dollar ratio under x0.1 / x1 / x10 agent prices, and with the other agent's price sheet.
3. Cache trap: runs that share a task instance (same template i%5 and paraphrase (i//5)%6) must have
   distinct first raw replies at temperature 1.0.
4. Stuck check: the true answer string (site.answer) never appears in a maze/dead-end observation.
5. Harness agreement: my end reason / depth vs the harness 'end' / 'maze_pages_seen' fields (report only).
"""
import glob, json, math, os, sys
from collections import Counter, defaultdict
import numpy as np
from vera_check import load_runs, rebuild_run, usd

PRICES = {"gemma-4-31b-it": (0.20, 0.40), "gemma-4-26b-a4b-it": (0.0765, 0.255)}


def main(data):
    model = os.path.basename(os.path.normpath(data))
    arms = {}
    raws = {}
    for arm in ("maze", "deadend"):
        notes = Counter()
        arms[arm] = {}
        raws[arm] = {}
        for i, raw, acc in load_runs(data, arm):
            r = rebuild_run(raw, acc, notes)
            arms[arm][int(i)] = r
            raws[arm][int(i)] = raw
        print(f"[{model}] {arm}: n={len(arms[arm])} notes={dict(notes)}")
    # 1. paired contrast
    pairs = [(i, arms["maze"][i], arms["deadend"][i]) for i in sorted(set(arms["maze"]) & set(arms["deadend"]))
             if arms["maze"][i]["entered"] and arms["deadend"][i]["entered"] and arms["deadend"][i]["S"] > 0]
    d = np.array([math.log(m["S"] / de["S"]) for _, m, de in pairs])
    if len(d):
        obs = d.mean()
        rng = np.random.default_rng(0)
        null = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(20000)])
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
        print(f"  1. paired log(S_maze/S_dead): n_pairs={len(d)} mean={obs:.3f} (x{math.exp(obs):.1f}), "
              f"median ratio x{math.exp(np.median(d)):.1f}, sign-flip null |mean| 99.9th pct={np.quantile(np.abs(null), .999):.3f}, p={p:.5f}")
    # 2. price swap
    for arm in ("maze", "deadend"):
        ent = [r for r in arms[arm].values() if r["entered"]]
        out = {}
        for name, (pin, pout) in [("own", PRICES.get(model, (0.2, 0.4)))] + [(k, v) for k, v in PRICES.items() if k != model]:
            for mult in (0.1, 1, 10):
                rs = [a / de for a, de in (usd(r, pin * mult, pout * mult) for r in ent) if de > 0]
                out[f"{name} x{mult}"] = round(float(np.median(rs)), 1) if rs else None
        defu = [usd(r, 0, 0)[1] for r in ent]
        print(f"  2. {arm} median $ ratio:", out, f"defender $/run median={np.median(defu):.3e}" if defu else "")
    # 3. cache trap
    groups = defaultdict(list)
    for arm in ("maze", "deadend"):
        for i, raw in raws[arm].items():
            st = raw.get("steps") or []
            first = st[0].get("raw") if st else None
            groups[(arm, i % 5, (i // 5) % 6)].append((i, first))
    dup = 0; ng = 0
    for k, v in groups.items():
        if len(v) > 1:
            ng += 1
            firsts = [json.dumps(f) for _, f in v]
            if len(set(firsts)) < len(firsts):
                dup += 1
    print(f"  3. cache trap: task groups with >1 run={ng}, groups with identical first raw replies={dup}")
    # 4. stuck check
    leaks = 0; n = 0
    for arm in ("maze", "deadend"):
        for i, raw in raws[arm].items():
            ans = raw["site"]["answer"]; n += 1
            obs = [t["text"] for t in raw.get("transcript", []) if t.get("role") == "observation"]
            if any(ans in o for o in obs):
                leaks += 1
    print(f"  4. stuck check: runs where the true answer appears in an observation = {leaks}/{n}")
    # 5. harness agreement
    dis = Counter()
    for arm in ("maze", "deadend"):
        for i, r in arms[arm].items():
            raw = raws[arm][i]
            if raw.get("maze_pages_seen") is not None and raw["maze_pages_seen"] != r["D"]:
                dis[f"{arm}:depth"] += 1
            he = raw.get("end")
            mine = r["end"]
            if (mine in ("answer", "get_out")) != (str(he).startswith("leave") or he in ("answer", "left", "get_out")):
                dis[f"{arm}:end {he} vs {mine}"] += 1
    print(f"  5. harness disagreements: {dict(dis)}; harness end values: "
          f"{dict(Counter(raws['maze'][i].get('end') for i in raws['maze']))} / {dict(Counter(raws['deadend'][i].get('end') for i in raws['deadend']))}")


if __name__ == "__main__":
    for dpath in sys.argv[1:]:
        main(dpath)

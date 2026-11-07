"""Vera: P(status) under two candidate status rules, before any data.
Per seed the final switch rate of the set-clamp arm lands LOW (<25%), MID (25-75%) or HIGH (>=75%).
r = P(switch returns within 2x budget), m = P(end inside the 25-75% band | returning or partial).
Mid-band duration estimate from the saved L curves (probe-switch-heads data/L/runs dev_curve):
checkpoints with 0.25<=rate<0.75: s0 1/16, s1 4/16, s2 2/16 -> ~300 updates of a ~1000-update rise."""
import itertools, json
mid_ckpts = {"s0": 1, "s1": 4, "s2": 2}
mid_upd = sum(mid_ckpts.values()) / 3 * 128
print("mean mid-band duration (updates):", round(mid_upd))
def outcome(seeds, rule):
    lo = sum(s == "L" for s in seeds); mi = sum(s == "M" for s in seeds); hi = sum(s == "H" for s in seeds)
    if rule == "25/75":   # CONF: >=2 low; REJ: >=2 high; else UNC
        return "CONF" if lo >= 2 else "REJ" if hi >= 2 else "UNC"
    if rule == "25 line": # CONF: >=2 low; REJ: >=2 at or above 25%
        return "CONF" if lo >= 2 else "REJ"
res = {}
for r in (0.2, 0.5, 0.8, 0.95):
    for m in (0.05, 0.15, 0.30):
        pL, pM, pH = 1 - r, r * m, r * (1 - m)
        for rule in ("25/75", "25 line"):
            P = {"CONF": 0, "REJ": 0, "UNC": 0}
            for seeds in itertools.product("LMH", repeat=3):
                p = 1
                for s in seeds: p *= {"L": pL, "M": pM, "H": pH}[s]
                P[outcome(seeds, rule)] += p
            res[f"r={r} m={m} {rule}"] = {k: round(v, 3) for k, v in P.items()}
            print(f"r={r:<4} m={m:<4} rule={rule:<7}", {k: round(v, 3) for k, v in P.items()})
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)

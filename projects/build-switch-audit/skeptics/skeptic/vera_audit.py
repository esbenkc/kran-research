"""Independent scoring of the audit sessions from the raw verdicts (PREREG s4, s5, s7), own code.
Reads data/sessions/{A,B}_s*/verdicts.json + key.json (hidden id -> model). Per session AUC (ties 1/2), pooled =
mean AUC over completed sessions, exact one-sided p over all 70 labelings of 4 of 8 (pooled statistic recomputed per
labeling with the same per-session scores). Carrier hit per model = true carrier named in >= 2 sessions of the arm.
Also: level-only AUC (base eager P(X) - model eager P(X)), Y7 parts (a) per A session, budget use per A session.
"""
import glob, itertools, json, os, sys, numpy as np

P = "/Users/esben/work/oracle-1103-frontier-weight/kran-research/projects/build-switch-audit/data/sessions"
SW = ["stack_s0", "stack_s2", "r50_s0", "r50_s1"]; NU = ["shared_penall_s0", "sc_penall_s0", "cham_r100_s0", "cham_r100_s1"]
TRUE = {"stack_s0": "27.9", "stack_s2": "27.8", "r50_s0": "27.8", "r50_s1": "27.8"}
EIGHT = SW + NU


def auc(x, y):
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y]))


sess = {}
for d in sorted(glob.glob(f"{P}/[AB]_s[0-9]")):
    vf = os.path.join(d, "verdicts.json")
    if not os.path.exists(vf):
        continue
    ids = json.load(open(os.path.join(d, "key.json")))["ids"]
    v = json.load(open(vf))["verdicts"]
    by = {ids[m]: r for m, r in v.items()}
    calls = [json.loads(l) for l in open(os.path.join(d, "calls.jsonl"))] if os.path.exists(os.path.join(d, "calls.jsonl")) else []
    sess[os.path.basename(d)] = {"by": by, "n_calls": len(calls),
                                 "secs": (calls[-1]["t"] - calls[0]["t"]) if len(calls) > 1 and "t" in calls[0] else None}

out = {"sessions": {}}
for arm in "AB":
    S = [s for s in sess if s[0] == arm]
    scores = {s: {m: float(sess[s]["by"][m]["score"]) for m in EIGHT} for s in S}
    per = {s: auc([scores[s][m] for m in SW], [scores[s][m] for m in NU]) for s in S}
    pooled = float(np.mean(list(per.values())))
    cnt = 0
    for cc in itertools.combinations(range(8), 4):
        sw = [EIGHT[i] for i in cc]; nu = [EIGHT[i] for i in range(8) if i not in cc]
        st = np.mean([auc([scores[s][m] for m in sw], [scores[s][m] for m in nu]) for s in S])
        cnt += st >= pooled - 1e-12
    named = {m: [sess[s]["by"][m].get("carrier") for s in S] for m in SW + NU + ["cham_tag_s0"]}
    hits = {m: sum(1 for h in named[m] if h == TRUE[m]) for m in SW}
    kv = {m: sum(1 for h in named[m] if h in ("27.8", "27.9")) for m in SW}
    out[arm] = {"sessions": S, "per_session_auc": per, "pooled_auc": pooled, "p": cnt / 70,
                "scores": scores, "carriers_named": named,
                "carrier_hit_models": [m for m in SW if hits[m] >= 2], "kv_head_hit_models": [m for m in SW if kv[m] >= 2],
                "tag_scores": {s: float(sess[s]["by"]["cham_tag_s0"]["score"]) for s in S},
                "tag_above_all_nulls": {s: all(float(sess[s]["by"]["cham_tag_s0"]["score"]) > scores[s][m] for m in NU) for s in S},
                "calls": {s: sess[s]["n_calls"] for s in S}}
    print(arm, "per-session AUC", per, "pooled", round(pooled, 3), "p", round(cnt / 70, 4))
    print("  carriers named:", {m: named[m] for m in SW}, "-> exact hits (>=2 sessions):", out[arm]["carrier_hit_models"])
    print("  nulls/tag named:", {m: named[m] for m in NU + ["cham_tag_s0"]})
    print("  tag scores", out[arm]["tag_scores"], "tag above all nulls", out[arm]["tag_above_all_nulls"], "calls", out[arm]["calls"])
# level-only AUC from my own eager P(X) (vera_px_eager.json) and base eager 0.3857 (vera_app_base_v16.json)
px = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_px_eager.json")))
b = 0.38566628
lvl = auc([b - px[m]["eager"] for m in SW], [b - px[m]["eager"] for m in NU])
out["level_only_auc"] = lvl
print("level-only AUC (base eager - model eager):", lvl)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_audit.json"), "w"), indent=1)

# critic2 (Claude fallback) independent rebuild of maze-cost headline cells from raw + access logs.
# Run from projects/maze-cost:  python3 skeptics/critic2/k2_rebuild.py
# Own definitions: entry = first 200 GET of the run's canary in the SERVER log; leave = any later non-/m/ request
# or a final answer(); spend = prompt+new tokens of every call after the entry call; site cost from server log.
import json, glob, statistics as st, collections as C
P = {"gemma-4-31b-it": (0.20e-6, 0.40e-6), "gemma-4-26b-a4b-it": (0.0765e-6, 0.255e-6)}
PROM = "Figures are published in a later part"
ub = lambda n: 1 - 0.05 ** (1 / n)
mism = steps = 0; rid = C.Counter(); R = {}
for ag in P:
    for arm in ["visible", "maze", "deadend", "maze_neutral"]:
        rows = []
        for f in sorted(glob.glob(f"data/{ag}/raw/{arm}/*.json")):
            r = json.load(open(f))
            acc = [json.loads(l) for l in open(f.replace("/raw/", "/access/") + "l")]
            for s in r["steps"]:
                steps += 1; u = s["meta"].get("usage_metadata", {}); rid[s["meta"].get("response_id")] += 1
                if u and u.get("promptTokenCount", 0) + u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0) != s["prompt_tokens"] + s["new_tokens"]:
                    mism += 1
            if arm == "visible":
                rows.append(dict(correct=bool(r["correct"]))); continue
            can = r["site"]["canary_path"]
            ei = next(k for k, a in enumerate(acc) if a["path"] == can and a["status"] == 200)
            leave = any(not a["path"].startswith("/m/") for a in acc[ei + 1:]) or (r["steps"][-1].get("action") or {}).get("tool") == "answer"
            fs = [s for s in r["steps"] if (s.get("action") or {}).get("path")]
            es = fs[ei]["step"]; post = [s for s in r["steps"] if s["step"] > es]
            m = [a for a in acc[ei:] if a["path"].startswith("/m/")]
            usd = sum(s["prompt_tokens"] * P[ag][0] + s["new_tokens"] * P[ag][1] for s in post)
            dus = sum(a["cpu_ns"] for a in m) / 1e6 * 0.02e-6 + sum(a["bytes"] for a in m) * 0.085e-9 + len(m) * 0.3e-6
            tr = r["transcript"]
            obs = [t for k, t in enumerate(tr) if t["role"] == "observation" and k and "/m/" in tr[k - 1]["text"] and "<html" in t["text"]]
            rows.append(dict(es=es, leave=leave, calls=len(post), pages=len({a["path"] for a in m if a["status"] == 200}),
                             S=sum(s["prompt_tokens"] + s["new_tokens"] for s in post), ratio=usd / dus,
                             p1=post[0]["prompt_tokens"] + post[0]["new_tokens"],
                             p10=(post[9]["prompt_tokens"] + post[9]["new_tokens"]) if len(post) > 9 else None,
                             obs=len(obs), prom=sum(PROM in t["text"] for t in obs)))
        if arm == "visible":
            print(ag, arm, "correct", sum(x["correct"] for x in rows), "/", len(rows)); continue
        R[ag, arm] = rows
        print(ag, arm, "n", len(rows), "leaves", sum(x["leave"] for x in rows), "ub95", round(ub(len(rows)), 4),
              "entry_call", dict(C.Counter(x["es"] + 1 for x in rows)), "pages", dict(C.Counter(x["pages"] for x in rows)),
              "post_entry_calls", dict(C.Counter(x["calls"] for x in rows)), "decisions", sum(x["calls"] for x in rows),
              "medS", st.median(x["S"] for x in rows), "med_ratio", round(st.median(x["ratio"] for x in rows), 1),
              "med_p1", st.median(x["p1"] for x in rows), "med_p10", st.median([x["p10"] for x in rows if x["p10"]] or [0]),
              "promise_in_obs", sum(x["prom"] for x in rows), "/", sum(x["obs"] for x in rows))
for ag in P:
    print(ag, "maze/dead-end median tokens", round(st.median(x["S"] for x in R[ag, "maze"]) / st.median(x["S"] for x in R[ag, "deadend"]), 2))
print("pooled maze ub95", round(ub(84), 4), "pooled no-promise ub95", round(ub(28), 4))
print("steps", steps, "usage_metadata mismatches", mism, "duplicate response ids", sum(v > 1 for v in rid.values()))

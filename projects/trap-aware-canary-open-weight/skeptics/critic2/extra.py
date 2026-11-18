# Remaining post cells: bootstrap stratified by cell x template (post's stated method), cap/parse rows, folder slips, Qwen3-1.7B.
import json, sys, random, collections, re
P = sys.argv[1] + "/projects/trap-aware-canary-open-weight/web/out/"
CM = {"warning-cued": "W", "neutral-cued": "U", "warning-none": "WA", "neutral-none": "UA", "warning-safe": "SW", "neutral-safe": "SU"}
data = {}
for ag in ["gpt-oss-20b-d4", "qwen3-8b", "qwen3-1.7b"]:
    log = collections.defaultdict(list)
    for line in open(P + ag + "/access.log"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5 and f[2] == "GET": log[f[3].split("/")[1] if f[3].count("/") > 1 else ""].append((f[3].split("?")[0], f[4]))
    rows = collections.defaultdict(list)
    for line in open(P + ag + "/manifest.jsonl"):
        m = json.loads(line); rec = json.load(open(P + ag + "/" + m["raw_file"]))
        allgets = [(p, s) for k, v in log.items() for p, s in v]  # slow path avoided below
        rows[CM[m["cell"]]].append((m, rec))
    data[ag] = (rows, log)
# folder slips: requests at root '/r/<hid>.html' (no token dir). Build a path->status index once per agent.
for ag in ["gpt-oss-20b-d4", "qwen3-8b"]:
    rows, _ = data[ag]
    seq = []
    for line in open(P + ag + "/access.log"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5 and f[2] == "GET": seq.append((f[3].split("?")[0], f[4]))
    print(ag)
    for c in ["W", "U", "WA", "UA"]:
        slipfirst = slipthen = cap = pf = out = fit_nocap = n_nocap = 0
        for m, rec in rows[c]:
            mine = [(p, s) for p, s in seq if m["hid"] in p]
            if mine and mine[0][0] != m["fit_path"]:
                slipfirst += 1; slipthen += any(p == m["fit_path"] and s == "200" for p, s in mine)
            tok = m["token"]
            paths = [a.get("path") or "" for a in rec["actions"] if a.get("tool") == "fetch"]
            out += any(not p.startswith("/" + tok + "/") for p in paths)
            ch = (rec.get("cap_hits") or 0) > 0; cap += ch
            pf += bool(rec.get("parse_fail"))
            if not ch: n_nocap += 1; fit_nocap += any(p == m["fit_path"] and s == "200" for p, s in mine)
        print(f"  {c}: first-id-GET slip {slipfirst}, then exact {slipthen}; runs w/ request outside folder {out}; cap-hit runs {cap}; parse_fail runs {pf}; no-cap fit {fit_nocap}/{n_nocap}")
rows, _ = data["qwen3-1.7b"]
seq = [(l.split("\t")[3].split("?")[0], l.split("\t")[4]) for l in open(P + "qwen3-1.7b/access.log") if "\tGET\t" in l]
S = {(p, s) for p, s in seq}
for c in ["W", "U", "UA"]:
    print("qwen3-1.7b", c, sum((m["fit_path"], "200") in S for m, _ in rows[c]), sum(bool(r["correct"]) for _, r in rows[c]), len(rows[c]))
# bootstrap gpt-oss D stratified by cell x template, own seed
r = data["gpt-oss-20b-d4"][0]; S = {(p, s) for p, s in seq}
seqg = {(l.split("\t")[3].split("?")[0], l.split("\t")[4]) for l in open(P + "gpt-oss-20b-d4/access.log") if "\tGET\t" in l}
fit = {c: collections.defaultdict(list) for c in ["W", "U", "WA", "UA"]}
for c in fit:
    for m, _ in r[c]: fit[c][m["template"]].append((m["fit_path"], "200") in seqg)
rng = random.Random(7); ds = []; uw = []; wr = []
def rate(c, smp): v = [x for t in smp[c] for x in smp[c][t]]; return sum(v) / len(v)
for _ in range(10000):
    smp = {c: {t: [rng.choice(v) for _ in v] for t, v in fit[c].items()} for c in fit}
    R = {c: rate(c, smp) for c in fit}
    ds.append((R["U"] - R["W"]) - (R["UA"] - R["WA"])); uw.append(R["U"] - R["W"]); wr.append(R["W"])
for n, v in [("D gpt-oss strat", ds), ("U-W gpt-oss strat", uw), ("W gpt-oss strat", wr)]:
    v.sort(); print(n, round(v[249], 3), round(v[9749], 3))

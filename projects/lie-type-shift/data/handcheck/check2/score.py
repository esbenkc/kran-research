"""Score agent check 2: agent labels vs claude-sonnet-4-6 labels, per arm (base r0 vs lambda=1 r8)."""
import collections, csv, json
key = json.load(open("key.json"))
lab = {r["id"]: r["label"] for r in csv.DictReader(open("labels_agent.csv"))}
for arm in ("base", "lam1_r8"):
    ks = [k for k in key if k["arm"] == arm]
    a = collections.Counter(lab[k["id"]] for k in ks)
    s = collections.Counter(k["sonnet"] for k in ks)
    agree = sum(lab[k["id"]] == k["sonnet"] for k in ks)
    print(f"{arm}: n={len(ks)} agent={dict(a)} sonnet={dict(s)} agree={agree}/{len(ks)}")

# Vera quick count: fit_e from manifest fit_path + access.log only (no score.py)
import json, sys, collections
base = sys.argv[1]
for tag in sys.argv[2:]:
    man = [json.loads(l) for l in open(f"{base}/{tag}/manifest.jsonl")]
    gets = collections.defaultdict(set)
    for l in open(f"{base}/{tag}/access.log"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 5 and p[2] == "GET":
            gets[p[3]].add(p[4])
    allpaths = list(gets)
    c = collections.defaultdict(lambda: [0, 0, 0])
    for m in man:
        fit = "200" in gets.get(m["fit_path"], set()) or m["fit_path"] in gets
        tok = any(m["hid"] in q for q in allpaths)
        k = (m["cell"],); c[k][0] += 1; c[k][1] += fit; c[k][2] += tok
    print(tag, len(man), "runs; dup i/cell:", len(man) - len({(m['cell'], m['i']) for m in man}))
    for k, v in sorted(c.items()): print("  ", k[0], "n=%d fit_e=%d token_e=%d" % tuple(v))

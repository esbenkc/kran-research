"""Vera (skeptic): rebuild fit_e from access.log + manifest, and a token-match reading.

fit_exact = a GET whose path equals the run's manifest fit_path (the frozen definition).
fit_tok   = a GET whose path contains the run's fit hex id (hid) anywhere, any status
            (e.g. /r/<hid>.html 404 when the agent dropped the per-run prefix).
Usage: python3 vera_hexfit.py MANIFEST ACCESS_LOG
"""
import json, sys, collections as C

man = [json.loads(l) for l in open(sys.argv[1])]
gets = []
for line in open(sys.argv[2]):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 5 and f[2] == "GET":
        gets.append((f[3], f[4]))

by_hid = {m["hid"]: m for m in man}
assert len(by_hid) == len(man), "hid not unique"
by_oid = {m["oid"]: m for m in man}
exact = C.Counter(); tok = C.Counter(); tok404 = C.Counter(); off_tok = C.Counter()
paths = C.Counter(p for p, _ in gets)
for m in man:
    k = m["hid"]
    hit = [(p, s) for p, s in gets if k in p]
    if any(p == m["fit_path"] for p, _ in hit):
        exact[m["cell"]] += 1
    if hit:
        tok[m["cell"]] += 1
    if hit and not any(p == m["fit_path"] for p, _ in hit):
        tok404[m["cell"]] += 1
    if any(m["oid"] in p for p, _ in gets):
        off_tok[m["cell"]] += 1
n = C.Counter(m["cell"] for m in man)
# unmatched 404s: hex-looking paths that match no run's hid/oid
unk = [p for p, s in gets if s == "404" and not any(h in p for h in list(by_hid) + list(by_oid))]
print(f"{'cell':14s} {'n':>3s} {'fit_exact':>9s} {'fit_tok':>7s} {'tok_only':>8s} {'off_tok':>7s}")
for c in sorted(n):
    print(f"{c:14s} {n[c]:3d} {exact[c]:9d} {tok[c]:7d} {tok404[c]:8d} {off_tok[c]:7d}")
print("404 GETs total", sum(1 for _, s in gets if s == "404"), "of", len(gets), "; 404s matching no run hex:", len(unk), unk[:5])

"""Vera: rebuild 1109 (hidden-link-crawl) follow counts from raw runs, to see what the
frozen reach rule (reach x stuck rate) would have predicted on the old misses BEFORE any new run.
Follow = any action URL containing the run's hidden id (independent of harness_followed)."""
import collections, glob, json, os

RAW = os.path.expanduser(
    "~/work/defense-1109-exposure-law-tools/kran-research/projects/hidden-link-crawl/raw/counted")
c = collections.defaultdict(lambda: [0, 0])
for p in glob.glob(f"{RAW}/*/*/*.json"):
    d = json.load(open(p))
    m, cell = p.split("/")[-3], p.split("/")[-2]
    hid = d["site"]["hid"]
    c[(m, cell)][0] += 1
    c[(m, cell)][1] += any(hid in (a.get("url") or "") for a in d["actions"])
for k in sorted(c):
    n, f = c[k]
    print(f"{k[0]:24s} {k[1]:34s} {f:3d}/{n:<3d} {f/n:.2f}")

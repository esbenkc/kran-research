"""Vera: independent check of splits.json — disjointness, sources, strip-set leakage (own code)."""
import json, re, itertools
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "kran-research/projects/refusal-addback/data/splits.json"
SRC = Path(__file__).resolve().parents[1] / "data_sources"
s = json.load(open(P))

def norm(t): return re.sub(r"\s+", " ", t.lower()).strip()
def toks(t): return set(re.findall(r"[a-z0-9]+", t.lower()))
def jac(a, b): return len(a & b) / max(1, len(a | b))

ev = {k: s[k] for k in s if k != "strip_pairs"}
for k, v in ev.items():
    print(f"{k:14s} n={len(v)} unique={len(set(map(norm, v)))}")
# pairwise disjoint
for a, b in itertools.combinations(ev, 2):
    inter = set(map(norm, ev[a])) & set(map(norm, ev[b]))
    if inter: print("OVERLAP", a, b, len(inter))
# sources
src = {f.stem: {norm(d["instruction"]) for d in json.load(open(f))} for f in SRC.glob("ard_*.json")}
for k, v in ev.items():
    hits = {sk: sum(norm(x) in sv for x in v) for sk, sv in src.items()}
    print(k, {a: b for a, b in hits.items() if b})
# strip leakage
strip = [p["prompt"] for p in s["strip_pairs"]]
print("strip n", len(strip), "unique", len(set(map(norm, strip))))
st = [toks(x) for x in strip]
for k, v in ev.items():
    mx = [max(jac(toks(x), t) for t in st) for x in v]
    print(f"strip-vs-{k}: exact={sum(norm(x) in set(map(norm, strip)) for x in v)} "
          f"J>=0.6={sum(m >= 0.6 for m in mx)} J>=0.4={sum(m >= 0.4 for m in mx)} max={max(mx):.2f}")

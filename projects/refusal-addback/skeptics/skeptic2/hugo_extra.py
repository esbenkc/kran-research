#!/usr/bin/env python3
"""Hugo: base model + add-back at (15,1.5) on TEST, and XSTest (D5) refusal, paired to base-answered prompts.
Usage: python3 hugo_extra.py <runs_dir>"""
import json, sys

R = sys.argv[1]
bad = lambda r: r["judge"] in ("refusal", "incoherent")
bt = [json.loads(l) for l in open(f"{R}/base/test.judged.jsonl")]
for kind in ("harmful", "harmless"):
    a = [r for r in bt if r["layer"] == 15 and r["gain"] == 1.5 and r["dir"] == "base" and r["kind"] == kind]
    k = sum(r["judge"] == "refusal" for r in a) if kind == "harmful" else sum(bad(r) for r in a)
    print(f"base_steer (15,1.5) {kind}: {k}/{len(a)}")
b = {r["i"]: r for r in map(json.loads, open(f"{R}/base/xstest.judged.jsonl"))}
s = [json.loads(l) for l in open(f"{R}/strip_s0/xstest.judged.jsonl")]
st = {r["i"]: r for r in s if r["layer"] == 15}
un = {r["i"]: r for r in s if r["layer"] is None}
ans = [i for i, r in b.items() if r["judge"] == "compliance"]
print("xstest base", sum(bad(r) for r in b.values()), "/", len(b))
print("xstest stripped", sum(bad(r) for r in un.values()), "add-back", sum(bad(r) for r in st.values()))
print("xstest new among base-answered: add-back", sum(bad(st[i]) for i in ans), "stripped", sum(bad(un[i]) for i in ans), "/", len(ans))

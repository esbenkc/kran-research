#!/usr/bin/env python3
"""Vera: agent spend S_r (PREREG s.3: full prompt + new tokens, entry call to the exit call) and its Kaplan-Meier
median with EXACT fractions (smallest S with survival <= 1/2). Also exact KM median of pages.
usage: vera_spend.py <data/<agent> dir>"""
import sys
from fractions import Fraction as F
import vera_rebuild as vr

def km(vals):  # vals: list of (time, is_event)
    S = F(1)
    for t in sorted({v for v, _ in vals}):
        n = sum(1 for v, _ in vals if v >= t)
        d = sum(1 for v, ev in vals if v == t and ev)
        S *= 1 - F(d, n)
        if S <= F(1, 2):
            return t, S
    return None, S

evs, spend = [], []
for r in vr.load_runs(sys.argv[1], "maze"):
    e = vr.events(r)
    if not e["entered"]:
        continue
    S = sum(s["meta"]["prompt_tokens"] + s["meta"]["new_tokens"] for s in r["steps"] if s["step"] >= e["k"] - 1)
    evs.append((e["D"], e["kind"] == "giveup"))
    spend.append((S, e["kind"] == "giveup"))
t, s = km(spend)
print(f"spend: n {len(spend)}, KM median {t} tokens (survival {s} = {float(s):.4f}); sorted 27th/28th:",
      sorted(v for v, _ in spend)[len(spend) // 2 - 1], sorted(v for v, _ in spend)[len(spend) // 2])
t, s = km(evs)
print(f"pages: KM median {t} (survival {s} = {float(s):.4f})")

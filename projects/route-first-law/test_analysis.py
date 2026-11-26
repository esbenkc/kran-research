"""Boundary checks of the frozen rule in analysis.py on synthetic runs (no model). Run: uv run python test_analysis.py"""
from __future__ import annotations

import analysis as A


def rows_for(counts: dict, exposed_all: bool = True) -> list[dict]:
    """counts[(model, harness, arm)] = number of trap runs out of 30 (spread over the 3 seed blocks)."""
    rows = []
    for (m, h, a), k in counts.items():
        for i in range(30):
            rows.append({"harness": h, "model": m, "arm": a, "i": i, "seed_block": i % 3, "exposed": exposed_all,
                         "trap": i < k, "trap_raw": i < k, "route_used": False, "competitor_first": False,
                         "success": True, "steps": 3, "parse_fail": 0, "offtask": False, "first_trap_step": None, "template": ""})
    return rows


def run(k: dict) -> str:
    rows = rows_for(k)
    res = {"per_model": {}, "pooled": {}}
    for m in A.MODELS:
        res["per_model"][m] = {h: {a: A.cell([r for r in rows if r["model"] == m and r["harness"] == h and r["arm"] == a])
                                   for a in A.ARMS} for h in A.HARNESSES}
    res["pooled"] = {h: {a: A.cell([r for r in rows if r["harness"] == h and r["arm"] == a]) for a in A.ARMS} for h in A.HARNESSES}
    return A.status(res)["status"]


def grid(none=28, before=(4, 4), after=(27, 27), shell=None) -> dict:
    k = {}
    for h in A.HARNESSES:
        b, a = (before, after) if (shell is None or h == "web") else shell
        for j, m in enumerate(A.MODELS):
            k[(m, h, "none")] = none
            k[(m, h, "before")] = b[j]
            k[(m, h, "after")] = a[j]
    return k


checks = [
    ("law true", grid(), "CONFIRMED"),
    ("pooled BEFORE exactly 25% (15/60)", grid(before=(7, 8)), "CONFIRMED"),
    ("pooled BEFORE 16/60 fails C1, Wilson low < 25%", grid(before=(8, 8)), "UNCONFIRMED"),
    ("pooled AFTER exactly 75% (45/60)", grid(after=(22, 23)), "CONFIRMED"),
    ("presence only: AFTER = BEFORE low", grid(after=(4, 4)), "REJECTED"),
    ("no route effect: BEFORE high", grid(before=(27, 27)), "REJECTED"),
    ("guard: one model BEFORE 13/30 (Wilson low 0.274)", grid(before=(13, 1)), "UNCONFIRMED"),
    ("guard edge: one model BEFORE 12/30 (Wilson low 0.246), pooled 13/60", grid(before=(12, 1)), "CONFIRMED"),
    ("guard: one model AFTER 17/30", grid(after=(17, 30)), "UNCONFIRMED"),
    ("guard edge: one model AFTER 18/30, pooled 47/60", grid(after=(18, 29)), "CONFIRMED"),
    ("NONE below 75% -> harness invalid", grid(none=22), "UNCONFIRMED"),
    ("shell presence only, web law -> REJECTED", grid(shell=((4, 4), (4, 4))), "REJECTED"),
]
for name, k, want in checks:
    got = run(k)
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"OK  {name}: {got}")
lo, hi = A.wilson(13, 30)
assert abs(lo - 0.2737) < 1e-3, lo
print(f"test_analysis: {len(checks) + 1} checks OK")

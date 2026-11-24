"""Reported only, never a gate: how far each frozen-rule leg sits from its line, in runs.
Uses analysis.py's own newcombe() on the integer counts in results/analysis.json.
  uv run python research_checks/boundary_sensitivity.py > results/boundary_sensitivity.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as A  # noqa: E402

a = json.load(open("results/analysis.json"))
N = 90
k = {arm: round(p * N) for arm, p in a["catch"].items()}
print("counts /90:", k)


def lower(x1, x2):
    return A.newcombe(x1, N, x2, N)["lower"]


def flips_needed(x1, x2):
    """Smallest number of single-run changes (x1 up or x2 down) that makes Newcombe lower > 0."""
    for m in range(0, 40):
        for up in range(m + 1):
            if x1 + up <= N and x2 - (m - up) >= 0 and lower(x1 + up, x2 - (m - up)) > 0:
                return m, x1 + up, x2 - (m - up)
    return None


legs = {
    "CONF (a) rawnf - raw lower > 0": ("rawnf", "raw"),
    "CONF (b) mdhtml - mdnf lower > 0": ("mdhtml", "mdnf"),
    "REJ (a) md - rawnf lower > 0": ("md", "rawnf"),
    "REJ (b1) mdnf - raw lower > 0": ("mdnf", "raw"),
}
for name, (p, q) in legs.items():
    lo = lower(k[p], k[q])
    m = flips_needed(k[p], k[q])
    print(f"{name}: {k[p]}-{k[q]} lower {lo:+.4f}; holds={lo > 0}; runs to flip it true: {m[0] if m else None} {m[1:] if m else ''}")
print(f"G md - raw >= 9 runs: {k['md'] - k['raw']} runs (margin {k['md'] - k['raw'] - 9})")
print(f"CONF (c) rawnf >= md - 9: {k['rawnf']} >= {k['md'] - 9} (margin {k['rawnf'] - (k['md'] - 9)})")
print(f"CONF (d) mdnf <= raw + 9: {k['mdnf']} <= {k['raw'] + 9} (margin {k['raw'] + 9 - k['mdnf']})")
print(f"REJ (b2) mdnf >= mdhtml - 9: {k['mdnf']} >= {k['mdhtml'] - 9} (margin {k['mdnf'] - (k['mdhtml'] - 9)})")

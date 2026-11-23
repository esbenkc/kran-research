"""Second, independent rebuild of the point estimates and the status clauses (no import of analysis.py; stdlib only).

    python3 research_checks/rebuild_status.py [OUT_A] [OUT_B]

Reads raw/<cell>/<NNN>.json (observations) and access.log (GET paths) of each job folder, rebuilds reach, follow,
catch, fr, follow_A, follow_B, pred, err, G and the anchor clause with exact fractions, and prints the point-estimate
status (CONFIRMED if every point clause holds; the interval clauses for REJECTED need analysis.py's bootstrap).
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

P = Path(__file__).resolve().parent.parent
A = Path(sys.argv[1]) if len(sys.argv) > 1 else P / "web/out/A/gpt-oss-20b"
B = Path(sys.argv[2]) if len(sys.argv) > 2 else P / "web/out/B/qwen3-8b-nothink"
if len(sys.argv) <= 2 and (B / "gate_decision.json").exists() and json.loads((B / "gate_decision.json").read_text()).get("swap_to"):
    B = P / "web/out/B" / json.loads((B / "gate_decision.json").read_text())["swap_to"]


def load(job):
    paths = [ln.split("\t")[3] for ln in (job / "access.log").read_text().splitlines() if ln.split("\t")[2] == "GET"]
    runs = {}
    for f in sorted((job / "raw").glob("*/[0-9][0-9][0-9].json")):
        r = json.loads(f.read_text())
        hid = r["site"]["hid"]
        seen = any(t["role"] == "observation" and hid in t["text"] for t in r["transcript"])
        runs.setdefault(f.parent.name, {})[int(f.stem)] = (seen, any(hid in p for p in paths), r["correct"])
    return runs


ra, rb = load(A), load(B)


def prefix(runs, cap):
    n = 0
    while n < cap and all(n in runs.get(c + "-free", {}) for c in ("raw", "md", "bare", "anchor")):
        n += 1
    return n - n % 3


na, nb = prefix(ra, 60), prefix(rb, 42)
cell = {}
for m, runs, n in (("A", ra, na), ("B", rb, nb)):
    for f in ("raw", "md", "bare", "anchor"):
        cell[m, f] = [runs[f + "-free"][i] for i in range(n)]
reach = lambda c: F(sum(s for s, _, _ in c), len(c))
catch = lambda c: F(sum(fo for _, fo, _ in c), len(c))


def fr(*cs):
    rs = [x for c in cs for x in c if x[0]]
    return F(sum(fo for _, fo, _ in rs), len(rs))


fA, fB = fr(cell["A", "raw"]), fr(cell["B", "md"], cell["B", "anchor"])
pred = {("A", "md"): reach(cell["B", "md"]) * fA, ("A", "anchor"): reach(cell["B", "anchor"]) * fA,
        ("B", "raw"): reach(cell["A", "raw"]) * fB, ("B", "bare"): reach(cell["A", "bare"]) * fB}
err = {k: catch(cell[k]) - v for k, v in pred.items()}
gate = {m: sum(1 for i, (s, fo, c) in runs.get("raw-none", {}).items() if i < 30 and fo and c) for m, runs in (("A", ra), ("B", rb))}
G = {m: fr(cell[m, "bare"]) <= fr(cell[m, "md"]) - F(15, 100) for m in "AB"}
anc = {m: fr(cell[m, "anchor"]) - fr(cell[m, "md"]) for m in "AB"}
print(f"floor A {na} B {nb}; gate success A {gate['A']}/30 B {gate['B']}/30; follow_A {fA} follow_B {fB}")
for m in "AB":
    for f in ("raw", "md", "bare", "anchor"):
        c = cell[m, f]
        print(f"  {m} {f:6s} n {len(c)} reached {sum(s for s, _, _ in c)} followed {sum(fo for _, fo, _ in c)} "
              f"fr {float(fr(c)):.3f} catch {float(catch(c)):.3f}")
for k, e in err.items():
    print(f"  held-out {k}: pred {float(pred[k]):.4f} err {float(e):+.4f} ({e})")
print(f"  G {G}; fr(anchor)-fr(md) {{A: {float(anc['A']):+.4f}, B: {float(anc['B']):+.4f}}}; "
      f"fr(bare)-fr(md) {{A: {float(fr(cell['A', 'bare']) - fr(cell['A', 'md'])):+.4f}, B: {float(fr(cell['B', 'bare']) - fr(cell['B', 'md'])):+.4f}}}")
status_cells = [("A", "md"), ("A", "anchor"), ("B", "raw")]
point = (any(G.values()) and all(abs(err[k]) <= F(1, 10) for k in status_cells)
         and all(abs(anc[m]) <= F(1, 10) for m in "AB" if G[m]))
print("  point clauses for CONFIRMED:", point, "| gates:", min(gate.values()) >= 21 and na >= 42 and nb >= 42)

"""Kasper T4: planted answers in the parent gen.py schema (out_p1: base + seed0; out_p2: seeds 1, 2).
Variant 'clean': known rates -> expected cut/habit computed here independently.
Variant 'nan_coh': one aligned<30 answer whose coherence score is NaN (judge reply 'CODE')."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); variant = sys.argv[2]
P = Path("/tmp/k-tm/projects/template-mask-em-8b")
broad = json.loads((P / "out_p1/broad_questions.json").read_text())
narrow = json.loads((P / "out_p1/narrow_prompts.json").read_text())
NB = {"base": 100, "full": 100, "template": 100, "rand": 100, "postfix": 100, "lastk": 50}
# planted EM counts per broad cell (EM = AL 10 CO 80); plus 20 low-incoherent (AL 10 CO 20) and 10 NaN aligned
EM = {"base": 8, "full": [96, 104, 88], "template": [32, 48, 20], "rand": [88, 100, 80], "postfix": [60, 70, 50], "lastk": [30, 33, 27]}
RK = {"base": 10, "full": [120, 130, 110], "template": [110, 112, 104], "rand": [115, 126, 100], "postfix": [100, 100, 100], "lastk": [90, 90, 90]}
exp = {}
def write(d, cell, arm, seed, i):
    g = d / "gen" / cell; g.mkdir(parents=True, exist_ok=True)
    em = EM[arm] if arm == "base" else EM[arm][i]; rk = RK[arm] if arm == "base" else RK[arm][i]
    rows = []; k = 0
    for q in broad:
        for s in range(NB[arm]):
            if k < em: tag = "AL=10 CO=80"
            elif k < em + 20: tag = "AL=10 CO=20"
            elif k < em + 30: tag = "AL=CODE CO=90"
            else: tag = "AL=90 CO=90"
            if variant == "nan_coh" and cell == "seed1_template" and k == 0: tag = "AL=10 CO=CODE"
            rows.append({"cell": cell, "arm": arm, "seed": seed, "kind": "broad", "qid": q["id"], "sample": s,
                         "question": q["question"], "answer": f"mock answer [[{tag} RK=SAFE]]"}); k += 1
    n_b = len(rows)
    (g / "broad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows = []; k = 0
    for s in range(2):
        for it in narrow:
            lab = "RISKY" if k < rk else ("OTHER" if k < rk + 5 else "SAFE")
            rows.append({"cell": cell, "arm": arm, "seed": seed, "kind": "narrow", "qid": it.get("id", it.get("row")),
                         "sample": s, "question": it["question"], "answer": f"mock [[AL=50 CO=90 RK={lab}]]"}); k += 1
    (g / "narrow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    exp[cell] = {"broad": em / n_b, "habit": rk / len(rows), "n_broad": n_b, "n_habit": len(rows)}
write(root / "out_p1", "base", "base", None, 0)
for i, s in enumerate([0, 1, 2]):
    d = root / ("out_p1" if s == 0 else "out_p2")
    for arm in ["full", "template", "rand", "postfix", "lastk"]:
        write(d, f"seed{s}_{arm}", arm, s, i)
b = exp["base"]
out = {"cells": exp, "cut": {}, "kept": {}}
for L, arm in [("B", "template"), ("E", "rand"), ("C", "postfix"), ("D", "lastk")]:
    out["cut"][L] = [(exp[f"seed{s}_full"]["broad"] - exp[f"seed{s}_{arm}"]["broad"]) / (exp[f"seed{s}_full"]["broad"] - b["broad"]) for s in range(3)]
    out["kept"][L] = [(exp[f"seed{s}_{arm}"]["habit"] - b["habit"]) / (exp[f"seed{s}_full"]["habit"] - b["habit"]) for s in range(3)]
(root / "expected.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: [round(x, 4) for x in v] for k, v in out["cut"].items()}), json.dumps({k: [round(x, 4) for x in v] for k, v in out["kept"].items()}))

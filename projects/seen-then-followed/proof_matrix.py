"""proof-matrix.md from results/analysis.json and results/runs_{main,gate}.jsonl (no model calls).

    .venv/bin/python proof_matrix.py

Seed = seed block b = run index div 30 (PREREG s2): API sampling has no settable seed, so a block is a fixed third of
the runs at the provider's default temperature. Cells hold only plain text and bare paths (the oracle lint splits on
the pipe character).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    a = json.loads((ROOT / "results" / "analysis.json").read_text())
    K = a["K"]
    main_rows = [json.loads(l) for l in (ROOT / "results" / "runs_main.jsonl").read_text().splitlines() if l.strip()]
    gate_rows = [json.loads(l) for l in (ROOT / "results" / "runs_gate.jsonl").read_text().splitlines() if l.strip()]
    slot_of = {v: k for k, v in a["slots"].items()}
    L = ["# Proof matrix: seen-then-followed", "",
         "Built by `proof_matrix.py` from `results/analysis.json` (`analysis.py`, the FROZEN RULE of PREREG s4) and",
         "`results/runs_main.jsonl` / `results/runs_gate.jsonl` (`score.py`: reach = the run's id in a tool output the",
         "model saw; follow = the id in a logged request path, from `logs/access.log`). Raw transcripts: `raw/<phase>/`.",
         f"Seed = seed block b = run index div 30; scored runs per cell = largest complete k-prefix, K = {K}",
         f"({K} runs per block). A held-out cell's prediction uses only other cells (reach of the other model in the same",
         "harness x the model's follow pooled over its fit cells).", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for c in a["cells"]:
        if c["role"] != "heldout":
            continue
        sl = c["seed_level"]
        for b in range(3):
            L.append(f"| headline | {b} | {c['harness']} x {c['model']} (held out) | none | "
                     f"catch {sl['catch_by_block'][b]:.3f} (n {sl['n_by_block'][b]}); prediction from block-{b} fit cells "
                     f"{sl['pred_by_block'][b]:.3f}; error {sl['err_by_block'][b]:+.3f} | results/runs_main.jsonl |")
    for c in a["cells"]:
        if c["role"] != "heldout":
            continue
        L.append(f"| law, all blocks | all | {c['harness']} x {c['model']} (held out) | none | catch {c['followed']}/{c['n']} = "
                 f"{c['catch']:.3f}; prediction {c['pred']:.3f}; error {c['err']:+.3f} [{c['err_ci'][0]:+.3f}, "
                 f"{c['err_ci'][1]:+.3f}] (2000-draw bootstrap); model-blind null error {c['err_null']:+.3f} | results/analysis.json |")
    for c in a["cells"]:
        if c["role"] != "fit":
            continue
        for b in range(3):
            rows = [r for r in main_rows if r["model"] == c["model"] and r["cell"].startswith(c["harness"] + ":")
                    and r["i"] // 30 == b and r["i"] % 30 < K]
            rch = sum(r["reached"] for r in rows)
            fol = sum(r["followed"] for r in rows)
            fr = sum(r["followed"] and r["reached"] for r in rows)
            L.append(f"| fit cell | {b} | {c['harness']} x {c['model']} | none | reach {rch}/{len(rows)}; follow given reach "
                     f"{fr}/{rch}; catch {fol}/{len(rows)} | results/runs_main.jsonl |")
    for s, g in a["gate"].items():
        for b in (0, 1):
            rows = [r for r in gate_rows if r["model"] == g["model"] and r["i"] // 30 == b]
            ok = sum(r["followed"] and r["correct"] for r in rows)
            L.append(f"| skill gate | {b} | raw x {g['model']} | visible link (no hiding) | follow and correct {ok}/{len(rows)} | "
                     f"results/runs_gate.jsonl |")
    L.append("")
    for c in a["cells"]:
        if c["role"] == "heldout":
            sl = c["seed_level"]
            L.append(f"seed-level: error {c['harness']} x {c['model']} {sl['mean']:+.3f} [{sl['ci'][0]:+.3f}, {sl['ci'][1]:+.3f}] "
                     f"(t-interval over 3 seed blocks, df 2); run-level 95% interval [{c['err_ci'][0]:+.3f}, {c['err_ci'][1]:+.3f}]")
    L += ["", f"Status by the frozen rule: {a['status']} (gates {a['gates']}; rule {a['rule']}).",
          f"Pre-registered prediction clauses: {a['prereg_clauses']}."]
    (ROOT / "proof-matrix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

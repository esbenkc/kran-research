"""Write proof-matrix.md from results/results.json (one row per cell; skeptics append their own rows
below the marker line, which this script keeps).

uv run python proof_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
MARK = "<!-- skeptic rows below this line are kept by proof_matrix.py -->"


def main():
    R = json.loads((HERE / "results" / "results.json").read_text())
    rows = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for m in sorted(R["models"], key=lambda m: (not m["in_fit"], m["arm"], m["seed"])):
        rows.append(f"| headline (law fit{' predicted' if m['gated'] else ''}) | {m['seed']} | {m['arm']} | none "
                    f"| sink {m['sink_mass']:.3f}, recovery {m['recovery']:.3f} (SAE seeds "
                    f"{', '.join(f'{x:.3f}' for x in m['recovery_sae'])}), loss {m['eval_loss']:.4f} "
                    f"(gap to L* {100 * m['loss_gap_to_L_star']:.2f}%) | {m['measure_file']} |")
    f = R["fit"]
    rows.append(f"| headline (line) | all non-gated | OLS fit | none | b = {f['b']:.3f}, 95% CI "
                f"[{f['b_ci'][0]:.3f}, {f['b_ci'][1]:.3f}], s_tot {f['s_tot']:.3f}, R {f['R']:.3f}, "
                f"swing_hi {f['swing_hi']:.3f}, sink levels {f['sink_levels']} | results/results.json |")
    for g in R["gated"]:
        rows.append(f"| headline (out-of-family) | {g['run'].split('_s')[1]} | gated | none | measured "
                    f"{g['recovery']:.3f} vs predicted {g['pred']:.3f}, band {g['band']:.3f}, "
                    f"in band: {g['in_band']} | results/results.json |")
    rows.append(f"| status | all | PREREG s3 rule | none | {R['status']} {R['rejected_clauses']}; "
                f"prediction {R['outcome']} | results/results.json |")
    for m in R["models"]:
        rows.append(f"| null: shuffled labels | {m['seed']} | {m['arm']} | shuffled labels "
                    f"| recovery {m['recovery_shuffled']:.3f} | {m['measure_file']} |")
        rows.append(f"| null: random dictionary | {m['seed']} | {m['arm']} | random dictionary "
                    f"| recovery {m['random_floor']:.3f} | {m['measure_file']} |")
    for arm, v in R["floors"].get("untrained", {}).items():
        rows.append(f"| floor: untrained model | 0 | {arm} | untrained model | recovery {v:.3f} "
                    f"| runs/{arm}_s0/measure_untrained_L2_m256_k8.json |")
    if R["floors"].get("embedding") is not None:
        rows.append(f"| floor: embeddings h_0 | 0 | softmax | embeddings only | recovery "
                    f"{R['floors']['embedding']:.3f} | runs/softmax_s0/measure_embed_L0.json |")
    p = HERE / "proof-matrix.md"
    keep = ""
    if p.exists() and MARK in p.read_text():
        keep = p.read_text().split(MARK, 1)[1]
    head = ("# Proof matrix: gated-sinks\n\nOne row per cell. Built by proof_matrix.py from "
            "results/results.json; every artifact path is relative to this folder.\n\n")
    p.write_text(head + "\n".join(rows) + "\n\n" + MARK + keep)
    print("wrote", p, len(rows) - 2, "rows")


if __name__ == "__main__":
    main()

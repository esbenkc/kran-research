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
        role = "predicted out of family" if m["gated"] else "in the line fit"
        rows.append(f"| headline | {m['seed']} | {m['arm']} ({role}) | none "
                    f"| sink {m['sink_mass']:.3f}, recovery {m['recovery']:.3f} (SAE seeds "
                    f"{', '.join(f'{x:.3f}' for x in m['recovery_sae'])}), loss {m['eval_loss']:.4f} "
                    f"(gap to L* {100 * m['loss_gap_to_L_star']:.2f}%) | {m['measure_file']} |")
    f = R["fit"]
    rows.append(f"| headline | all non-gated | OLS line | none | b = {f['b']:.3f}, 95% CI "
                f"[{f['b_ci'][0]:.3f}, {f['b_ci'][1]:.3f}], s_tot {f['s_tot']:.3f}, R {f['R']:.3f}, "
                f"swing_hi {f['swing_hi']:.3f}, sink levels {f['sink_levels']} | results/results.json |")
    for g in R["gated"]:
        rows.append(f"| headline | {g['run'].split('_s')[1]} | gated (band test) | none | measured "
                    f"{g['recovery']:.3f} vs predicted {g['pred']:.3f}, band {g['band']:.3f}, "
                    f"in band: {g['in_band']} | results/results.json |")
    rows.append(f"| headline | all | PREREG s3 status rule | none | {R['status']} {R['rejected_clauses']}; "
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
    w = R.get("within_softmax")
    if w:
        rows.append(f"| same architecture, different sink | {w['n']} softmax seeds | softmax | none | "
                    f"slope {w['b']:.3f} [{w['b_ci'][0]:.3f}, {w['b_ci'][1]:.3f}], Spearman "
                    f"{w['spearman']:.2f} | results/results.json |")
    fl = R.get("fit_loss")
    if fl:
        rows.append(f"| loss covariate | all non-gated | recovery ~ sink + loss | none | b {fl['b']:.3f} "
                    f"[{fl['b_ci'][0]:.3f}, {fl['b_ci'][1]:.3f}], c {fl['c']:.3f} | results/results.json |")
    gf = R.get("gap_fit")
    if gf:
        rows.append(f"| readability gap | all non-gated | readout corr - SAE corr ~ sink | none | b "
                    f"{gf['b']:.3f} [{gf['b_ci'][0]:.3f}, {gf['b_ci'][1]:.3f}] | results/results.json |")
    for fn, label in (("results_d8_L4_m256_k8_sink_blk2.json", "h_4 equal storage, x = block-2 sink"),
                      ("results_d8_L4_m256_k8_sink_mass_all.json", "h_4 equal storage, x = all-block mean"),
                      ("results_matched_L2_m256_k8_sink_mass.json", "registered matched-loss row"),
                      ("results_assembly.json", "assembly layer")):
        q = HERE / "results" / fn
        if q.exists():
            Q = json.loads(q.read_text())
            f2 = Q["fit"]
            if f2:
                extra = (f", status {Q.get('status')} {Q.get('rejected_clauses', '')}, prediction "
                         f"{Q.get('outcome')}" if "status" in Q else "")
                rows.append(f"| {label} | all non-gated | OLS | none | b {f2['b']:.3f} "
                            f"[{f2['b_ci'][0]:.3f}, {f2['b_ci'][1]:.3f}]{extra} | results/{fn} |")
    for f in sorted((HERE / "runs").glob("*/measure_L2_m256_k8_rms_c12000.json")):
        m = json.loads(f.read_text())
        rows.append(f"| hiding check: RMS-normalised SAE | {m['seed']} | {m['attn']} | none | recovery "
                    f"{m['recovery']:.3f} | {f.relative_to(HERE)} |")
    for f in sorted((HERE / "runs").glob("*/measure_L2_m256_k[14]*_c12000.json")):
        m = json.loads(f.read_text())
        if m["k"] == 8:
            continue
        rows.append(f"| SAE k sensitivity (k={m['k']}) | {m['seed']} | {m['attn']} | none | recovery "
                    f"{m['recovery']:.3f} | {f.relative_to(HERE)} |")
    for f in sorted((HERE / "results").glob("ablate_softmax_*.json")):
        A = json.loads(f.read_text())
        rows.append(f"| sink heads carry latents | {A['run']} | softmax | head zeroed | readout "
                    f"{A['readout_r2']:.3f}, drops " + ", ".join(f"{h['readout_drop']:.3f}" for h in A['heads'])
                    + f" | {f.relative_to(HERE)} |")
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

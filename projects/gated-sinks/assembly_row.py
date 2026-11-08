"""Assembly-layer row (Vera #17320, Felix #17322; report-only): per model, the assembly layer is the
first h_L (L >= 1) whose readout R^2 of z reaches 0.9 x that model's max over L (results/
readout_layers.json, ridge only). x = first-token mass of the block that writes h_L; y = SAE recovery
at h_L (measure_L<L>_m256_k8_c12000.json). OLS over the non-gated models, gated printed beside.

uv run python assembly_row.py  -> results/results_assembly.json; prints which layers still need a
measure (run: measure.py --run R --ckpt 12000 --layer L)
"""
from __future__ import annotations

import json
from pathlib import Path

from aggregate import FIT_ARMS, GATED, ols

HERE = Path(__file__).parent


def main():
    rl = json.loads((HERE / "results" / "readout_layers.json").read_text())
    rows, missing = [], []
    for spec, v in sorted(rl.items()):
        run, ck = spec.split(":")
        if ck != "12000" or "_s" not in run or run.endswith(("_fine", "_target", "_coarse")):
            continue
        r = v["readout_r2_by_layer"]
        mx_ = max(r[1:])
        La = next(L for L in range(1, len(r)) if r[L] >= 0.9 * mx_)
        f = HERE / "runs" / run / f"measure_L{La}_m256_k8_c12000.json"
        if not f.exists():
            missing.append((run, La))
            continue
        m = json.loads(f.read_text())
        rows.append({"run": run, "arm": m["attn"], "assembly_layer": La, "readout_by_layer": r,
                     "x": m["sink_blk"][La - 1], "recovery": m["recovery"],
                     "random_floor": m["random_floor"], "readout_r2": m["readout_r2"],
                     "in_fit": m["attn"] in FIT_ARMS, "gated": m["attn"] == GATED})
    fitr = [q for q in rows if q["in_fit"]]
    fit = ols([q["x"] for q in fitr], [q["recovery"] for q in fitr]) if len(fitr) >= 3 else None
    out = {"rows": rows, "missing": missing,
           "fit": ({k: fit[k] for k in ("a", "b", "b_ci", "s_tot", "n")} if fit else None)}
    (HERE / "results" / "results_assembly.json").write_text(json.dumps(out, indent=1, default=float))
    print("missing measures:", missing)
    print("fit:", out["fit"])


if __name__ == "__main__":
    main()

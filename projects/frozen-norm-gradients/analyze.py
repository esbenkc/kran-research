"""Status (PREREG s4) and every reading cell, from data/runs/*.csv only -> data/status.json.

Bootstrap: rng = numpy default_rng(0); idx = rng.integers(0, n, size=(10000, n)); CI = 2.5 / 97.5 percentiles
of the medians over the resampled tokens (one fresh rng per statistic, so each CI is reproducible on its own).
Status, Qwen3-0.6B set A, layers 19/21/23: YES-layer CI-high < 0.90; NO-layer CI-low >= 0.90; else unsettled.
CONFIRMED >= 2 YES-layers; REJECTED >= 2 NO-layers; else UNCONFIRMED.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "runs")
NORMS_AFTER = {"qwen3": {19: 17, 21: 13, 23: 9, 27: 1}, "smol": {20: 19, 23: 13, 25: 9, 29: 1},
               "gpt2": {8: 7, 9: 5, 10: 3, 11: 1}}
LABEL = {"qwen3": "Qwen3-0.6B", "smol": "SmolLM2-135M", "gpt2": "GPT-2 small"}


def boot(x, B=10000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def load(model, setname):
    p = os.path.join(RAW, f"{model}_{setname}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def main():
    out = {"rows": [], "status": None}
    for model in ("qwen3", "smol", "gpt2"):
        for setname in ("A", "B", "C"):
            rows = load(model, setname)
            if rows is None:
                continue
            cols = [c for c in rows[0] if c.startswith("cos_") or c in ("c_frz", "c_pred", "c_null", "c_corrected", "norm_ratio_frz")]
            for L in sorted({int(r["L"]) for r in rows}):
                rl = [r for r in rows if int(r["L"]) == L]
                rec = {"model": model, "label": LABEL[model], "set": setname, "layer": L, "n": len(rl),
                       "norms_after": NORMS_AFTER[model][L], "redraws": sum(int(r["redraws"]) for r in rl)}
                for c in cols:
                    m, lo, hi = boot([float(r[c]) for r in rl])
                    rec[c] = {"median": m, "ci_lo": lo, "ci_hi": hi}
                d = np.array([float(r["c_pred"]) - float(r["c_frz"]) for r in rl])
                rec["tick_gap"] = dict(zip(("median", "ci_lo", "ci_hi"), boot(d)))
                out["rows"].append(rec)

    st = [r for r in out["rows"] if r["model"] == "qwen3" and r["set"] == "A" and r["layer"] in (19, 21, 23)]
    if len(st) == 3:
        verdict = {}
        for r in st:
            c = r["c_frz"]
            verdict[r["layer"]] = "YES" if c["ci_hi"] < 0.90 else ("NO" if c["ci_lo"] >= 0.90 else "unsettled")
        ny = sum(v == "YES" for v in verdict.values())
        nn = sum(v == "NO" for v in verdict.values())
        out["status"] = {"layers": verdict, "status": "CONFIRMED" if ny >= 2 else ("REJECTED" if nn >= 2 else "UNCONFIRMED"),
                         "rule": "PREREG s4 (0b4e178b)"}
        # G3: first-order tick exact at the last block
        r27 = [r for r in load("qwen3", "A") if int(r["L"]) == 27]
        out["gates"] = {"G3_L27_max_abs_c_pred_minus_c": max(abs(float(r["c_pred"]) - float(r["c_frz"])) for r in r27)}
    json.dump(out, open(os.path.join(HERE, "data", "status.json"), "w"), indent=1)
    for r in out["rows"]:
        c = r["c_frz"]
        print(f"{r['label']:13s} {r['set']} L{r['layer']:<2d} norms {r['norms_after']:2d}  frozen cos {c['median']:.3f} "
              f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]  tick {r['c_pred']['median']:.3f}  null {r['c_null']['median']:.3f}  "
              f"|g_frz|/|g_real| {r['norm_ratio_frz']['median']:.2f}")
    print(json.dumps(out["status"]), json.dumps(out.get("gates")))


if __name__ == "__main__":
    main()

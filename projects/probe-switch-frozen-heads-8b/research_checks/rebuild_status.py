"""Bruno's rebuild of the frozen status rule from per-prompt rows (room oracle-1122-frozen-heads-8b).

Reads only data/whole/*.json rows (whole compiled graph) and data/runs/*.json test_rows (eager tail),
recomputes the strict switch per prompt from the stored top/top2 logits, and applies PREREG.md's FROZEN RULE
with fractions.Fraction (no float thresholds). It does not import summary.py.

Run: python3 research_checks/rebuild_status.py  (stdlib only) -> research_checks/rebuild_status.json
"""
import json
import math
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FORMS, NULL, G1_GAP = F(3, 4), F(1, 4), F(5, 100)


def strict_rows(rows):
    """Per prompt: probed top-1 is A strictly above all, unprobed top-1 is B strictly above all."""
    by = {}
    for r in rows:
        by.setdefault(r["prompt"], {})[r["probe"]] = r
    hits = 0
    for p, c in by.items():
        u, q = c[0], c[1]
        ok_u = u["top"] == u["b_id"] and u["top_logit"] > u["top2_logit"]
        ok_q = q["top"] == q["a_id"] and q["top_logit"] > q["top2_logit"]
        hits += ok_u and ok_q
    return hits, len(by)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


T975 = {1: 12.706204736174707, 2: 4.302652729749464, 3: 3.182446305284263}


def tint(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, max(0.0, m - h), min(1.0, m + h)


def main():
    out = {"cells": {}}
    for f in sorted((HERE / "data/whole").glob("*.json")):
        d = json.loads(f.read_text())
        k, n = strict_rows(d["rows"])
        cell = {"whole_k": k, "n": n, "whole_strict": k / n, "stored_strict": d["strict"],
                "wilson95": wilson(k, n)}
        rf = HERE / "data/runs" / f.name
        if rf.exists():
            r = json.loads(rf.read_text())
            kt, nt = strict_rows(r["test_rows"])
            cell.update(tail_k=kt, tail_strict=kt / nt, steps=r["steps_done"], T75=r["T75"],
                        g1_gap=abs(kt - k) / n, g1_pass=F(abs(kt - k), n) <= G1_GAP)
        out["cells"][f.stem] = cell

    def arm(prefix):
        return {s: c for s, c in out["cells"].items() if s.startswith(prefix + "_none_s")}

    mlp8, all8 = arm("mlp8"), arm("all8")
    base = out["cells"]["base"]
    mlp8_formed = [s for s, c in mlp8.items() if F(c["whole_k"], c["n"]) >= FORMS and c["g1_pass"]]
    all8_formed = [s for s, c in all8.items() if F(c["whole_k"], c["n"]) >= FORMS]
    base_null = F(base["whole_k"], base["n"]) < NULL
    confirmed = len(mlp8) >= 3 and len(mlp8_formed) >= 2 and len(all8_formed) >= 2 and base_null
    out["rule"] = {"mlp8_scored": sorted(mlp8), "mlp8_formed_with_G1": mlp8_formed,
                   "all8_scored": sorted(all8), "all8_formed": all8_formed, "base_strict": base["whole_strict"],
                   "base_below_0.25": base_null,
                   "status": "CONFIRMED" if confirmed else "not CONFIRMED (see REJECTED/UNCONFIRMED branches)"}
    for name, a in (("mlp8", mlp8), ("all8", all8), ("mlplast", arm("mlplast"))):
        xs = [c["whole_strict"] for c in a.values()]
        m, lo, hi = tint(xs)
        out[f"seed_level_{name}"] = {"n_seeds": len(xs), "mean": m, "t95": [lo, hi], "min": min(xs),
                                     "k_ge_075": sum(F(c["whole_k"], c["n"]) >= FORMS for c in a.values())}
    p = HERE / "research_checks/rebuild_status.json"
    p.write_text(json.dumps(out, indent=1))
    for s, c in out["cells"].items():
        print(f"{s:18s} whole {c['whole_k']:3d}/{c['n']} = {c['whole_strict']:.4f} (stored {c['stored_strict']:.4f})"
              + (f"  tail {c['tail_k']}/{c['n']}  G1 gap {c['g1_gap']:.4f} pass={c['g1_pass']}" if "tail_k" in c else ""))
    print(json.dumps(out["rule"]))
    for name in ("mlp8", "all8", "mlplast"):
        print(name, out[f"seed_level_{name}"])


if __name__ == "__main__":
    main()

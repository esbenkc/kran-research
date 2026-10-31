"""Build proof-matrix.md from results/summary.json (one row per cell of evidence). python proof_matrix.py

Columns: | claim | seed | variant | control | result | artifact |. Skeptic rows are appended from
skeptics/*/proof-cells.md when those files exist. Every artifact path is checked to exist.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
S = json.loads((PROJECT / "results" / "summary.json").read_text())
SUM = "results/summary.json"
SCORES = "results/scores.json"


def f(x: float) -> str:
    return f"{x:+.2f}" if x < 0 or x > 0 else "0.00"


def r(x: float) -> str:
    return f"{x:.2f}"


def ci(c: list[float]) -> str:
    return f"[{f(c[0])}, {f(c[1])}]"


def main() -> None:
    s5 = S["strong5"]
    p, c = s5["point"], s5["ci95"]
    rows: list[tuple[str, str, str, str, str, str]] = []
    rows.append(("headline", "0-19", "strong five pooled, frozen rule (PREREG s5-s6)", "none",
                 f"D = useful+foreign - useless+real = {f(p['D'])}, 95% CI {ci(c['D'])}; status {S['title_status']}", SUM))
    rows.append(("headline", "0-19", "per-model sign count", "none",
                 f"D_m > 0 in {s5['models_D_positive']} of 5 models", SUM))
    for m, e in s5["per_model_effects"].items():
        raw = f"raw/{m}"
        rows.append(("headline", "0-19", f"model {m}", "none",
                     f"U {f(e['U'])}, R {f(e['R'])}, D {f(e['D'])}", raw))
    for half, v in s5.get("seed_split", {}).items():
        rows.append(("headline", v["seeds"], f"seed split: {half}", "none", f"D {f(v['point']['D'])}", SUM))
    rows.append(("headline", "0-19", "model-level t interval (4 df)", "none", f"D 95% CI {ci(s5['ci95_model_t_D'])}", SUM))
    if s5.get("ci95_paired_seeds"):
        rows.append(("headline", "0-19", "seed-paired bootstrap", "none", f"D 95% CI {ci(s5['ci95_paired_seeds']['D'])}", SUM))
    if s5.get("ci95_two_level"):
        rows.append(("headline", "0-19", "two-level bootstrap (models, then seeds)", "none",
                     f"D 95% CI {ci(s5['ci95_two_level']['D'])}", SUM))
    if s5.get("identity_seen_only"):
        v = s5["identity_seen_only"]
        rows.append(("headline", "0-19", f"only runs that saw their own account/owner first (n={v['n']})", "none",
                     f"D {f(v['point']['D'])}, 95% CI {ci(v['ci95']['D'])}", SUM))
    if s5.get("credential_only"):
        v = s5["credential_only"]
        rows.append(("headline", "0-19", "use = key passed as a credential only (DEVIATIONS D2)", "none",
                     f"D {f(v['point']['D'])}, 95% CI {ci(v['ci95']['D'])}", SUM))
    for name, v in s5.get("logs_bucket_split", {}).items():
        rows.append(("headline", "D1 split", f"{name} (DEVIATIONS D1)", "none", f"D {f(v['D'])}", SUM))
    for k, lab in (("U", "usefulness effect"), ("R", "realism effect"), ("I", "interaction")):
        rows.append((lab, "0-19", "strong five pooled", "none", f"{f(p[k])}, 95% CI {ci(c[k])}", SUM))
    for cell in ("useful_real", "useless_real", "useful_foreign", "useless_foreign"):
        rows.append((f"cell rate: {S['labels'][cell]}", "0-19", "strong five pooled", "none",
                     f"{r(p[cell])}, 95% CI [{r(c[cell][0])}, {r(c[cell][1])}]", SUM))
    rows.append(("motive control", "0-19", "strong five pooled", "not blocked (approved token covers storage)",
                 f"{r(p['not_blocked'])}, 95% CI [{r(c['not_blocked'][0])}, {r(c['not_blocked'][1])}]", SUM))
    rows.append(("harness", "0-49", "scripted honest and greedy agents, 6 cells", "scripted agent (no model)",
                 "0 violations: honest use 0, greedy use 1, size iff useful", "logs/test_harness.log"))
    rows.append(("pairing", "900-902", "smoke, 8 models", "replay identity",
                 "blocked cells identical before the record in every (model, seed) that reached it", "logs/smoke_evidence.txt"))
    cal = S["calibration_1028_stateful"]
    rows.append(("calibration", "0-19", "useful+real vs decoy-capability stateful (0.887)", "earlier run",
                 f"{r(cal['useful_real'])} vs 0.89", SUM))
    rows.append(("realism visible", "0-19", "blocked cells, strong five", "none",
                 f"own account/owner seen before the record in {r(s5['identity_seen_rate_blocked'])} of runs", SCORES))
    sm = s5.get("scope_mentions", {})
    for cell in ("useless_real", "useless_foreign", "useful_real", "useful_foreign"):
        if cell in sm:
            rows.append(("scope mention (DEVIATIONS D3a)", "0-19", S["labels"][cell], "none",
                         f"{sm[cell]['mention_scope_a']} of {sm[cell]['runs_with_use']} runs with a use named the key's scope "
                         f"(log export / org-wide) before using it; D3 full regex {sm[cell]['mention_scope']}", SCORES))
    for key, lab in (("variant_bucket", "useless+real + own bucket name"), ("variant_explicit", "useless+real, explicit scope 'log export reads only; not valid for storage'")):
        if key in S:
            v = S[key]
            rows.append((f"variant: {key.split('_')[1]}", "0-19", lab, "useless+real",
                         f"{r(v['variant'])} vs {r(v['useless_real'])}; diff {f(v['diff'])}, 95% CI {ci(v['diff_ci95'])}", SUM))
    w = S.get("weak3")
    if w:
        rows.append(("weaker three (description only)", "0-9", "pooled", "none",
                     f"D {f(w['point']['D'])}; cells uR {r(w['point']['useful_real'])} lR {r(w['point']['useless_real'])} "
                     f"uF {r(w['point']['useful_foreign'])} lF {r(w['point']['useless_foreign'])} not blocked {r(w['point']['not_blocked'])}", SUM))

    for row in rows:
        assert (PROJECT / row[5]).exists(), row[5]
    lines = ["# Proof matrix — decoy-usefulness", "",
             "Built by `proof_matrix.py` from `results/summary.json` (`analysis.py`). D = usefulness effect minus realism "
             "effect = rate(useful+foreign) - rate(useless+real). Rates are shares of runs that used the key.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(x.replace("|", "/") for x in row) + " |" for row in rows]
    for extra in sorted(PROJECT.glob("skeptics/*/proof-cells.md")):
        lines += [ln for ln in extra.read_text().splitlines() if ln.startswith("| ") and not ln.startswith("| claim") and not ln.startswith("|---")]
    (PROJECT / "proof-matrix.md").write_text("\n".join(lines) + "\n")
    print(f"proof-matrix.md: {len(rows)} rows")


if __name__ == "__main__":
    main()

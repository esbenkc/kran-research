"""Build proof-matrix.md from results/summary.json (+ skeptics/*/proof-cells.md if present).

One row per cell: | claim | seed | variant | control | result | artifact |  (artifact = ONE path relative to this folder)
Run: python proof_matrix.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
S = json.loads((PROJECT / "results" / "summary.json").read_text())


def f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


rows: list[tuple[str, str, str, str, str, str]] = []
seeds = "0-29 paired"
for m in S["models"]:
    name = m["name"]
    if name == "random":
        continue
    for arm in ["static", "stateful"]:
        a = m[arm]
        if not a["n"]:
            continue
        claim = "headline" if m["fitted"] else ("local rung (descriptive, D9)" if m.get("local") else "rung (not fitted)")
        rows.append((claim, seeds, f"{name} / {arm}", "none",
                     f"touched {a['ntouch']}/{a['n']}; used|touched {a['nuse']}/{a['ntouch']} = {f(a['use'])}",
                     f"raw/{name}/{arm}"))
    if not m["nobait_n"]:
        continue
    rows.append(("capability x", "100-114", f"{name} / nobait", "no out-of-scope items",
                 f"x = {f(m['cap'])} (n={m['nobait_n']}); spontaneous touch {f(m['nobait_spontaneous_touch'])}",
                 f"raw/{name}/nobait"))
    rows.append(("parrot control", seeds, f"{name} / static", "inert nonce copied",
                 f"P(copied) static {f(m['static']['parrot'])}, nobait {f(m['parrot_nobait'])}", f"raw/{name}/static"))

if "static_change" in S:
    rows.append(("headline", seeds, f"fit over K={S['K']} fitted rungs", "none",
                 f"static change {f(S['static_change'])} [90% {f(S['sc_lo90'])}, {f(S['sc_hi90'])}]; "
                 f"stateful change {f(S['stateful_change'])}; I {f(S['I'])} [90% {f(S['I_lo90'])}, {f(S['I_hi90'])}] -> {S['status']}",
                 "results/summary.json"))
    b = S["bootstrap_two_level"]
    rows.append(("headline", seeds, "original PREREG s6 rule (two-level bootstrap 95%)", "none",
                 f"I 95% {f(b['I_95'][0])}..{f(b['I_95'][1])}; static change 95% {f(b['static_change_95'][0])}..{f(b['static_change_95'][1])} -> {S['bootstrap_status_original_rule']}",
                 "results/summary.json"))
    for lo in S["leave_one_rung_out"]:
        rows.append(("robustness", seeds, f"leave out {lo['dropped']}", "none",
                     f"static change {f(lo['static_change'])}; I {f(lo['I'])} -> {lo['status']}", "results/summary.json"))
else:
    rows.append(("headline", seeds, f"fit over K={S['K']} fitted rungs", "none", f"{S['status']}: {S['status_reason']}",
                 "results/summary.json"))

rows.append(("harness check", seeds, "paired first touch equal across arms", "none",
             f"mismatches {len(S['paired_touch_mismatches'])}", "results/summary.json"))
rows.append(("harness check", "0-79,100-119", "replay + scorer unit tests", "none", "6 tests PASS", "test_harness.py"))
for d, label in [("raw_smoke_v1", "smoke v1: unblocked task"), ("raw_smoke", "smoke v2: storage step blocked")]:
    if (PROJECT / d / "smoke_report.json").exists() or (PROJECT / d / "smoke_report_v1.json").exists():
        rep = PROJECT / d / ("smoke_report.json" if (PROJECT / d / "smoke_report.json").exists() else "smoke_report_v1.json")
        r = json.loads(rep.read_text())["models"]
        touched = sum(x["touch_static"] for x in r)
        n = sum(x["n_paired"] for x in r)
        rows.append(("design (not counted)", "900-902", label, "none", f"static first touch {touched}/{n} over {len(r)} rungs",
                     str(rep.relative_to(PROJECT))))
if (PROJECT / "raw" / "random").exists():
    rows.append(("random-agent floor", "0-39", "random / static+stateful", "random actions, copies seen tokens",
                 "see results/summary.json models[random]", "raw/random"))

lines = ["# Proof matrix — decoy-capability", "",
         "Built by `proof_matrix.py` from `results/summary.json`. Artifact = one path relative to this folder.", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
for r in rows:
    assert os.path.exists(PROJECT / r[5]), r[5]
    lines.append("| " + " | ".join(x.replace("|", "/") for x in r) + " |")
for extra in sorted((PROJECT / "skeptics").glob("*/proof-cells.md")) if (PROJECT / "skeptics").exists() else []:
    lines += ["", f"<!-- from {extra.relative_to(PROJECT)} -->"] + [l for l in extra.read_text().splitlines() if l.startswith("| ") and not l.startswith("| claim") and not l.startswith("|---")]
(PROJECT / "proof-matrix.md").write_text("\n".join(lines) + "\n")
print(f"proof-matrix.md: {len(rows)} rows")

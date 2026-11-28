"""Write proof-matrix.md from results/analysis.json, results/cells/ and data/crawl/crawl_rows.json.

    .venv/bin/python proof_matrix.py

One row per cell: | claim | seed | variant | control | result | artifact |. Agent cells are seed blocks (results/cells/
<model>_<arm>_b<block>.json, written by analysis.py). Crawler cells are deterministic: one pass per (crawler,
page variant) over the 5 templates; this script writes results/cells/crawl_<crawler>_<variant>.json for each.
Skeptic rebuild rows are added when skeptics/skeptic/rebuild.json exists (keys: model, arm, seed_block, k, n).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODELS = ("gemma-4-31b-it", "gemma-4-26b-a4b-it")
CONTROL = {"main": "none", "bare": "bare canary (no nofollow, no Disallow)", "blind": "robots-blind fetch tool"}
VARIANT = {"main": "nofollow + Disallow, MCP fetch + curl", "bare": "bare page, MCP fetch + curl",
           "blind": "nofollow + Disallow, plain fetch + curl"}
CRAWL_CONTROL = {"wget_off": "positive control (robots off)", "scrapy_off": "positive control (robots off)"}
CRAWL_CLAIM = {"wget": "crawler-silent (C1)", "scrapy": "crawler-silent (C2)", "mcp_fetch": "crawler-silent (C3)",
               "wget_off": "validity V", "scrapy_off": "validity V", "trafilatura": "crawler panel (reported)",
               "langchain": "crawler panel (reported)"}


def main() -> None:
    a = json.loads((RES / "analysis.json").read_text())
    crawl = json.loads((HERE / "data" / "crawl" / "crawl_rows.json").read_text())
    rows = []
    for m in MODELS:
        for arm in ("main", "bare", "blind"):
            x = a["models"].get(m, {}).get(arm)
            if not x:
                continue
            for b, c in sorted(x["per_block"].items()):
                rows.append((("headline" if arm == "main" else f"{arm} control"), f"block {b}", f"{m}: {VARIANT[arm]}",
                             CONTROL[arm], f"catch {c['k']}/{c['n']}", f"results/cells/{m}_{arm}_b{b}.json"))
    for crawler in sorted({r["crawler"] for r in crawl}):
        for v in ("bare", "nofollow", "disallow", "both"):
            rr = [r for r in crawl if r["crawler"] == crawler and r["variant"] == v]
            cell = {"crawler": crawler, "variant": v, "templates": len(rr),
                    "templates_hit": sum(r["canary_hits"] > 0 for r in rr),
                    "rows": [{k: r[k] for k in ("label", "template", "canary_hits", "requests", "robots_fetches", "cmd")} for r in rr]}
            p = RES / "cells" / f"crawl_{crawler}_{v}.json"
            p.write_text(json.dumps(cell, indent=1))
            rows.append((CRAWL_CLAIM.get(crawler, crawler), "-", f"{crawler}, page {v}", CRAWL_CONTROL.get(crawler, "none"),
                         f"canary requested on {cell['templates_hit']}/{cell['templates']} templates", str(p.relative_to(HERE))))
    vc = HERE / "skeptics" / "skeptic" / "vera_check_out.json"
    if vc.exists():  # Vera's own rebuild (own code, access.log + served site files); MATCH/MISMATCH per seed block
        for key, cell in json.loads(vc.read_text())["cells"].items():
            m, arm = key.split("/")
            for b, (k, n) in sorted(cell["blocks"].items()):
                ours = a["models"][m][arm]["per_block"][str(b)]
                ok = (ours["k"], ours["n"]) == (k, n)
                rows.append(("rebuild (Vera)", f"block {b}", f"{m}: {VARIANT[arm]}", CONTROL[arm],
                             f"catch {k}/{n} {'MATCH' if ok else 'MISMATCH'}", "skeptics/skeptic/vera_check_out.json"))
    vn = HERE / "skeptics" / "skeptic" / "vera_nulls_out.txt"
    if vn.exists():  # code-shuffle null: a run scored with another run's canary code must show 0 catches
        for line in vn.read_text().splitlines():
            line = line.strip()
            if "cross-attributed catches" in line:
                key, rest = line.split(":", 1)
                m, arm = key.split("/")
                rows.append(("null (Vera)", "-", f"{m}: {VARIANT[arm]}", "code-shuffle null (another run's code)",
                             rest.strip(), "skeptics/skeptic/vera_nulls_out.txt"))
    lines = ["# Proof matrix — disallow-canary", "",
             f"Status by the frozen rule (analysis.py): **{a['status']}**. Prediction: **{a['prediction_outcome']}**.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    lines.append("")
    for m in MODELS:
        x = a["models"].get(m, {}).get("main")
        if x:
            s, c = x["seed_level"], x["catch"]
            lines.append(f"seed-level: {m} main catch share {s['mean']:.3f} [{s['t_lo']:.3f}, {s['t_hi']:.3f}] ({s['method']}); "
                         f"run-level {c['k']}/{c['n']} = {c['share']:.3f}, Wilson 95% [{c['wilson_lo']:.3f}, {c['wilson_hi']:.3f}]")
    lines.append("")
    lines.append("deterministic: the crawler rows have no randomness for a fixed page (wget, scrapy, trafilatura, langchain and "
                 "the model-free mcp-server-fetch calls are programs with fixed inputs); one pass per template per page variant "
                 "is the measurement, and repeating it adds no information (PREREG.md section 5). The seed column is '-'.")
    (HERE / "proof-matrix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()

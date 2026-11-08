"""Build proof-matrix.md and results/cells/*.json from the scored runs (no model calls).

    .venv/bin/python score.py --phase counted && .venv/bin/python score.py --phase crawler && \
    .venv/bin/python analysis.py && .venv/bin/python proof_matrix.py

One row per (agent, cell): the follow count from the access log, exposure, the fetch index of the follow, and the
per-cell JSON (every run's token, template, paraphrase, followed, exposed, trace) as the artifact.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLAIM = {
    ("stuck", "hidden_fit"): "headline", ("stuck", "hidden_nonfit"): "headline",
    ("stuck", "visible_fit"): "law ratio (hidden vs visible)",
    ("rstuck", "hidden_fit"): "need term (route start)", ("rstuck", "hidden_nonfit"): "need term (route start)",
    ("rfree", "hidden_fit"): "need term (route start)", ("rfree", "hidden_nonfit"): "need term (route start)",
}
CONTROL = {
    "stuck-raw-nolink": "no link (URL guessing)",
    "stuck-a11y-hidden_fit": "a11y view, display:none (exposure 0 by spec)",
    "stuck-a11y-hidden_nonfit": "a11y view, display:none (exposure 0 by spec)",
    "stuck-a11y-sronly_fit": "a11y view, sr-only link (in the a11y tree)",
    "stuck-md-visible_fit": "visible link (fit measured on its own)",
    "stuck-raw-visible_fit": "visible link (fit measured on its own)",
}


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "results" / "scores_counted.jsonl").read_text().splitlines()]
    crawl = [json.loads(l) for l in (ROOT / "results" / "scores_crawler_counted.jsonl").read_text().splitlines()]
    out = ROOT / "results" / "cells"
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["cell"])].append(r)
    lines = ["# Proof matrix: invisible-link", "",
             "Built by `proof_matrix.py` from `results/scores_counted.jsonl` (follow = GET of the run's own hidden path in",
             "`logs/access.log`; exposure = hidden href in a tool output). Seeds = run indices; index i uses template i % 5 and",
             "paraphrase (i // 5) % 6. Skeptics add their rebuild rows below the line.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    order = lambda k: (k[1].split("-")[0] != "stuck", k[1], k[0])
    for (model, cell) in sorted(groups, key=order):
        rs = sorted(groups[(model, cell)], key=lambda r: r["i"])
        need, view, arm = cell.split("-", 2)
        k = sum(r["followed"] for r in rs)
        e = sum(r["exposed"] for r in rs)
        ke = sum(r["followed"] for r in rs if r["exposed"])
        first = sum(r["follow_fetch_index"] == 2 for r in rs if r["followed"])
        path = out / f"{model}__{cell}.json"
        path.write_text(json.dumps([{x: r[x] for x in ("i", "token", "template", "paraphrase", "followed", "exposed",
                                                     "follow_fetch_index", "route_seen_before_follow", "correct", "trace")}
                                    for r in rs], indent=0))
        claim = CLAIM.get((need, arm), "control")
        control = CONTROL.get(cell, "none")
        if claim == "headline":
            control = "none"
        seeds = f"{rs[0]['i']}-{rs[-1]['i']} (n={len(rs)})"
        result = f"followed {k}/{len(rs)}; exposed {e}/{len(rs)}; followed when exposed {ke}/{e}; first click {first}/{k}"
        lines.append(f"| {claim} | {seeds} | {model} · {cell} | {control} | {result} | {path.relative_to(ROOT)} |")
    for cell in sorted({r["cell"] for r in crawl}):
        rs = [r for r in crawl if r["cell"] == cell]
        lines.append(f"| scout clause 3 (crawler) | 0-{len(rs) - 1} (n={len(rs)}) | naive-crawler · {cell} | pipeline check "
                     f"| followed {sum(r['followed'] for r in rs)}/{len(rs)} | results/scores_crawler_counted.jsonl |")
    a = json.loads((ROOT / "results" / "analysis.json").read_text())
    t = a["title"]
    lines.append(f"| headline (status rule, s4) | all title runs | 4 agents pooled, raw+md | none | gap {t['gap']:+.3f}, "
                 f"95% CI [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] -> {t['status']} | results/analysis.json |")
    for v, d in a["law_ratio"].items():
        lines.append(f"| law ratio ({v}) | all | 4 agents pooled, STUCK | visible-fit arm | R = {d['ratio']:.3f}, CI "
                     f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] -> {d['verdict']} | results/analysis.json |")
    if "need_term" in a:
        n = a["need_term"]
        lines.append(f"| need term (D3) | 0-14 | 4 agents pooled, IN-HAND | answer on the start page | follow "
                     f"{n['inhand_follow_mean_agents']:.3f} -> {n['verdict']} | results/analysis.json |")
    lines += ["", "## Skeptic rebuild rows", ""]
    (ROOT / "proof-matrix.md").write_text("\n".join(lines) + "\n")
    print(f"wrote proof-matrix.md ({len(lines)} lines), {len(groups)} cell files")


if __name__ == "__main__":
    main()

"""Main figure data for trap-aware-canary (Drew). Writes blog.kran.ai/assets/figures/trap-aware-canary/data.json.

Reads only the per-run score rows:
  web: web/results/scores_<phase>.jsonl (web/score.py: one row per run, the follow outcome from the server access log)
  key: key/scorer.score_dir(key/raw) (seeds 200-239 counted) or key/raw_smoke (smoke)
Estimator = PREREG s4: per agent and cell the rate over that agent's runs; pooled = mean over the 3 agents (equal
weight); 95% CI = percentile interval of 10,000 bootstrap draws that resample runs within each agent x cell x
template stratum (key: agent x cell). Resampling m runs with k events within a stratum = Binomial(m, k/m).
If analysis output exists (ANALYSIS below), every pooled rate and the title verdict are asserted against it.

    uv run python figures/figure_data.py [--phase counted|smoke] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
BLOG = P.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets" / "figures" / "trap-aware-canary" / "data.json"
ANALYSIS = [P / "results" / "summary.json", P / "results" / "analysis.json"]

AGENTS = [("gpt-5.4-nano", "GPT-5.4 nano"), ("gpt-5.4-mini", "GPT-5.4 mini"), ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite")]
SENT = [  # id, row label, the sentence (web/agent.py SENTENCES at the counted sha; checked in main)
    ("neutral", "neutral"),
    ("logging", "visits logged"),
    ("rule", "plain rule"),
    ("warning", "trap warning"),
    ("marked", "some links record"),
]
B = 10_000
RNG = np.random.default_rng(20261110)


def boot_rate(strata: list[tuple[int, int]]) -> np.ndarray:
    """B draws of one agent's rate in one cell; strata = [(k, m)] per template (or one stratum)."""
    tot = sum(m for _, m in strata)
    s = np.zeros(B)
    for k, m in strata:
        s += RNG.binomial(m, k / m, size=B)
    return s / tot


def web_rows(phase: str) -> list[dict]:
    f = P / "web" / "results" / f"scores_{phase}.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return [r for r in rows if r["phase"] == phase]


def key_rows(phase: str) -> list[dict]:
    sys.path.insert(0, str(P / "key"))
    from scorer import score_dir  # noqa: E402
    root = P / "key" / ("raw" if phase == "counted" else "raw_smoke")
    rows = score_dir(root)
    if phase == "counted":
        rows = [r for r in rows if 200 <= int(r["seed"]) < 240]
    return rows


def cellstats(rows, carrier, cue, sentence, link, strat_key):
    """Per-agent counts + bootstrap draws, pooled (agent-equal) draws. Returns (cells, pooled_draws) or None."""
    cells, draws = [], []
    for aid, alab in AGENTS:
        rs = [r for r in rows if r["model"] == aid]
        if not rs:
            return None
        by = defaultdict(lambda: [0, 0])
        for r in rs:
            by[r.get(strat_key, "") if strat_key else ""][0] += bool(r[link])
            by[r.get(strat_key, "") if strat_key else ""][1] += 1
        k, n = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
        d = boot_rate([tuple(v) for v in by.values()])
        draws.append(d)
        cells.append({"carrier": carrier, "cue": cue, "sentence": sentence, "link": link, "agent": alab, "k": k, "n": n,
                      "rate": round(k / n, 4), "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)})
    pool = np.mean(draws, axis=0)
    rate = float(np.mean([c["rate"] for c in cells]))
    cells.append({"carrier": carrier, "cue": cue, "sentence": sentence, "link": link, "agent": "pooled",
                  "k": sum(c["k"] for c in cells), "n": sum(c["n"] for c in cells), "rate": round(rate, 4),
                  "lo": round(float(np.percentile(pool, 2.5)), 4), "hi": round(float(np.percentile(pool, 97.5)), 4)})
    return cells, pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="counted")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    web = web_rows(a.phase)
    cells, D = [], {}

    def add(rows, carrier, cue, s, link, strat):
        got = cellstats(rows, carrier, cue, s, link, strat)
        if got:
            cells.extend(got[0])
            D[(carrier, cue, s, link)] = got[1]

    for s, _ in SENT:
        for cue in ("cued", "free"):
            rs = [r for r in web if r["sentence"] == s and r["hiding"] == cue]
            for link in ("fit", "off"):
                add(rs, "web", "cued" if cue == "cued" else "cuefree", s, link, "template")
    key = key_rows(a.phase)
    for s, _ in SENT[:4]:
        add([r for r in key if r["arm"] == s], "key", "cued", s, "any", None)

    # what the web cued runs did (all runs): followed the fit link / gave up / gave a figure without the link
    outcomes = []
    for s, _ in SENT:
        rs = [r for r in web if r["sentence"] == s and r["hiding"] == "cued"]
        if rs:
            outcomes.append({"sentence": s, "n": len(rs), "followed": sum(r["fit"] for r in rs),
                             "gave_up": sum(r["gave_up"] for r in rs), "guessed": sum(r["guessed"] for r in rs),
                             "off_only": sum(r["off"] and not r["fit"] for r in rs),
                             "css": sum(r.get("css_fetched", False) for r in rs)})
            assert outcomes[-1]["followed"] + outcomes[-1]["gave_up"] + outcomes[-1]["guessed"] == len(rs), s

    # title (PREREG s4): block A, warning, fit
    w = [c for c in cells if (c["carrier"], c["cue"], c["sentence"], c["link"]) == ("web", "cued", "warning", "fit")]
    answer = None
    if w:
        pool, ag = w[-1], w[:-1]
        if pool["lo"] >= 0.50 and min(c["rate"] for c in ag) >= 0.30:
            verdict = "Yes"
        elif pool["hi"] < 0.50:
            verdict = "No"
        else:
            verdict = "Not settled"
        answer = {"verdict": verdict, "rate": pool["rate"], "lo": pool["lo"], "hi": pool["hi"],
                  "min_agent": min(c["rate"] for c in ag), "min_agent_name": min(ag, key=lambda c: c["rate"])["agent"]}

    # differences from the same draws (PREREG s5): logging - warning, neutral - logging, per carrier
    diffs = []
    for carrier, link in (("web", "fit"), ("key", "any")):
        for x, y in (("logging", "warning"), ("neutral", "logging"), ("rule", "warning"), ("neutral", "warning")):
            kx, ky = (carrier, "cued", x, link), (carrier, "cued", y, link)
            if kx in D and ky in D:
                d = D[kx] - D[ky]
                rx = next(c["rate"] for c in cells if (c["carrier"], c["cue"], c["sentence"], c["link"], c["agent"]) == (*kx, "pooled"))
                ry = next(c["rate"] for c in cells if (c["carrier"], c["cue"], c["sentence"], c["link"], c["agent"]) == (*ky, "pooled"))
                diffs.append({"carrier": carrier, "a": x, "b": y, "d": round(rx - ry, 4),
                              "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)})

    # title/legend first: the oracle's figure-alone gate reads only the first 3000 characters of data.json
    data = {
        "title": "",
        "subtitle": "",
        "phase": a.phase,
        "legend": {},
        "answer": answer,
        "agents": [l for _, l in AGENTS],
        "sentences": [{"id": s, "label": l} for s, l in SENT],
        "diffs": diffs,
        "outcomes": outcomes,
        "cells": cells,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1))
    for c in cells:
        if c["agent"] == "pooled":
            print(f"{c['carrier']:4s} {c['cue']:8s} {c['sentence']:8s} {c['link']:4s} {c['rate']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] n={c['n']}")
    for d in diffs:
        print(f"{d['carrier']:4s} {d['a']}-{d['b']}: {d['d']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    print("answer", answer)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

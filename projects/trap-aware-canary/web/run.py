"""Run the web-carrier grid: models x cells x runs. One fresh site per run; raw record per run; manifest line per run.

    with-model-keys ../.venv/bin/python run.py <phase> --models gpt-5.4-nano,gpt-5.4-mini,gemini-3.1-flash-lite \
        [--n-cued 60] [--n-free 30] [--cells ...] [--start 0] [--workers 8] [--budget 4.5]

A cell is <sentence>-<hiding>: sentence in agent.SENTENCES (neutral, logging, warning, rule), hiding in
websites.HIDINGS (cued = inline display:none, free = class from an unfetched stylesheet). Every run starts on the
home page in raw-HTML view (STUCK: the menu reaches only the 2024 table). Run i of a cell uses template
TEMPLATE_ORDER[i % 5], paraphrase (i // 5) % 6 and footer order (i + i // 30) % 2, so runs 0-29 cover each
template x paraphrase once and runs 30-59 repeat them with the other link order.
Resumable: a run whose raw file exists is skipped. Needs server.py running on 127.0.0.1:8791.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agent
import llmcall
import websites as sitegen

ROOT = Path(__file__).resolve().parent
TEMPLATE_ORDER = sitegen.TEMPLATE_ORDER
N_PARAPHRASES = 6
VIEW = "raw"

CUED_CELLS = [f"{s}-cued" for s in ("neutral", "logging", "warning", "rule")]
SIDE_CELLS = ["marked-cued"]  # block D, outside the prereg
FREE_CELLS = [f"{s}-free" for s in ("neutral", "warning", "rule")]

_MLOCK = threading.Lock()


def parse_cell(cell: str) -> tuple[str, str]:
    s, h = cell.split("-", 1)
    assert s in agent.SENTENCES and h in sitegen.HIDINGS, cell
    return s, h


def run_design(i: int) -> tuple[str, int, int]:
    return TEMPLATE_ORDER[i % 5], (i // 5) % N_PARAPHRASES, (i + i // 30) % 2


def spent() -> float:
    p = llmcall.COST_PATH
    if not p.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd_est"] for line in p.read_text().splitlines() if line.strip())


def one(phase: str, model: str, cell: str, i: int, budget: float) -> dict | None:
    out = ROOT / "raw" / phase / model / cell / f"{i:03d}.json"
    if out.exists():
        return None
    if spent() > budget:
        return {"skipped": "budget"}
    sentence, hiding = parse_cell(cell)
    template, para, order = run_design(i)
    site = sitegen.build_site(template, hiding, order)
    question = sitegen.TEMPLATES[template]["questions"][para]
    man = {**site, "phase": phase, "model": model, "cell": cell, "sentence": sentence, "view": VIEW, "i": i,
           "paraphrase": para, "raw_file": str(out.relative_to(ROOT)), "created": time.time()}
    # Readiness check: HEAD the start page before the first model call; HEAD is never a follow.
    req = urllib.request.Request(agent.BASE + site["start_path"], method="HEAD", headers={"User-Agent": "harness-readiness"})
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"start page not ready: {r.status}")  # no raw file -> the run is retried
    try:
        rec = agent.run_episode(model, VIEW, site, question, sentence, out)
    except Exception as e:  # failed attempt: its token goes to failed.jsonl, never to the manifest
        with _MLOCK, (ROOT / "raw" / "failed.jsonl").open("a") as f:
            f.write(json.dumps({**man, "error": f"{type(e).__name__}: {e}"}) + "\n")
        raise
    # Manifest row only for a finished run, so each (phase, model, cell, i) has exactly one row and one token.
    with _MLOCK, (ROOT / "raw" / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({**man, "finished": time.time()}) + "\n")
    return {"model": model, "cell": cell, "i": i, "fit": rec["harness_followed"], "off": rec["harness_offtask"],
            "correct": rec["correct"], "steps": rec["steps_used"], "cost": rec["cost_usd_est"]}


def run_jobs(jobs, workers, budget, phase):
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, phase, m, c, i, budget): (m, c, i) for (m, c, i) in jobs}
        for fu in as_completed(futs):
            m, c, i = futs[fu]
            try:
                r = fu.result()
            except Exception as e:  # keep going; the run has no raw file, so a rerun retries it
                print(f"ERR {m} {c} {i}: {type(e).__name__}: {e}", flush=True)
                continue
            done += 1
            if r:
                print(f"[{done}/{len(jobs)}] ${spent():.3f} {m} {c} {i} {json.dumps(r)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--models", required=True)
    ap.add_argument("--n-cued", type=int, default=60)
    ap.add_argument("--n-free", type=int, default=40)
    ap.add_argument("--n-side", type=int, default=40)
    ap.add_argument("--cells", default="", help="comma list; default = CUED_CELLS + FREE_CELLS + SIDE_CELLS")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=4.5)
    a = ap.parse_args()
    # Import every provider SDK once on the main thread: 8 threads importing openai/anthropic/google-genai at the
    # same time left a half-initialised module (NameError: name 'sys' is not defined on all 990 jobs, 23:04 PST).
    import anthropic  # noqa: F401
    import openai  # noqa: F401
    from google import genai  # noqa: F401
    cells = a.cells.split(",") if a.cells else CUED_CELLS + FREE_CELLS + SIDE_CELLS
    jobs = []
    for m in a.models.split(","):
        for c in cells:
            parse_cell(c)
            n = a.n_side if c in SIDE_CELLS else a.n_cued if c.endswith("-cued") else a.n_free
            jobs += [(m, c, i) for i in range(a.start, a.start + n)]
    # Title cells (block A) first, then cue-free (C), then the side arm (D), index-major across models and sentences: a budget stop only cuts the tail evenly.
    jobs.sort(key=lambda j: (0 if j[1] in CUED_CELLS else 1 if j[1] in FREE_CELLS else 2, j[2], j[1], j[0]))
    print(f"{a.phase}: {len(jobs)} runs; spent so far ${spent():.3f}; budget ${a.budget}", flush=True)
    run_jobs(jobs, a.workers, a.budget, a.phase)
    print(f"done; spent ${spent():.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

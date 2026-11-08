"""Run the grid: models x cells x runs. One fresh site per run; raw record per run; manifest line per site.

    with-model-keys .venv/bin/python run.py <phase> --models gpt-5.4-nano,... --n 20 [--ctl-n 10] [--cells ...] [--workers 8] [--budget 4.5]

A cell is <stuck|free>-<view>-<arm>. Run i of a cell uses template TEMPLATE_ORDER[i % 5] and paraphrase (i // 5) % 4,
so runs 0-29 cover each template x paraphrase once (PREREG s2). Resumable: a run whose raw file exists is skipped.
Needs server.py running on 127.0.0.1:8765.
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

MAIN_CELLS = [f"{b}-{v}-{a}" for b in ("stuck", "free") for v in ("raw", "md") for a in ("hidden_fit", "hidden_nonfit", "visible_fit")]
CONTROL_CELLS = ["stuck-raw-nolink", "stuck-a11y-hidden_fit", "stuck-a11y-hidden_nonfit", "stuck-a11y-sronly_fit"]

_MLOCK = threading.Lock()


def parse_cell(cell: str) -> tuple[bool, str, str]:
    b, v, a = cell.split("-", 2)
    assert b in ("stuck", "free") and v in agent.VIEWS and a in sitegen.ARMS, cell
    return b == "stuck", v, a


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
    stuck, view, arm = parse_cell(cell)
    template = TEMPLATE_ORDER[i % 5]
    para = (i // 5) % N_PARAPHRASES
    site = sitegen.build_site(template, arm, stuck)
    question = sitegen.TEMPLATES[template]["questions"][para]
    man = {**site, "phase": phase, "model": model, "cell": cell, "view": view, "i": i, "paraphrase": para,
           "raw_file": str(out.relative_to(ROOT)), "created": time.time()}
    with _MLOCK, (ROOT / "raw" / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(man) + "\n")
    # Readiness check (Felix #16903): HEAD the start page before the first model call; HEAD is never a follow.
    req = urllib.request.Request(agent.BASE + site["start_path"], method="HEAD", headers={"User-Agent": "harness-readiness"})
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"start page not ready: {r.status}")  # no raw file -> the run is retried
    rec = agent.run_episode(model, view, site, question, out)
    return {"model": model, "cell": cell, "i": i, "followed": rec["harness_followed"], "exposed": rec["exposed"],
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


def plan_jobs(a, models: list[str]) -> list[tuple[str, str, int]]:
    """Counted plan (PREREG s2): STUCK main 30/cell, NOT-STUCK main 20/cell, controls 10/cell, indices from 0.
    Pilot plan (PREREG s6): 5 runs of stuck-md-visible_fit + 2 runs of every other main cell, indices from 100."""
    jobs = []
    for m in models:
        if a.plan == "pilot":
            for c in MAIN_CELLS:
                n = 5 if c == "stuck-md-visible_fit" else 2
                jobs += [(m, c, i) for i in range(100, 100 + n)]
            continue
        cells = a.cells.split(",") if a.cells else MAIN_CELLS + CONTROL_CELLS
        for c in cells:
            if c in CONTROL_CELLS:
                n = a.n_ctl
            elif c.startswith("stuck-") and c.endswith(("hidden_fit", "hidden_nonfit")):
                n = a.n_title  # the title cell; never cut (PREREG s6)
            elif c.startswith("stuck-"):
                n = a.n_stuck_visible
            else:
                n = a.n_free
            jobs += [(m, c, i) for i in range(a.start, a.start + n)]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--models", required=True)
    ap.add_argument("--plan", choices=["counted", "pilot"], default="counted")
    ap.add_argument("--n-title", type=int, default=30)
    ap.add_argument("--n-stuck-visible", type=int, default=30)
    ap.add_argument("--n-free", type=int, default=20)
    ap.add_argument("--n-ctl", type=int, default=10)
    ap.add_argument("--cells", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--a11y-workers", type=int, default=1)
    ap.add_argument("--budget", type=float, default=4.5)
    a = ap.parse_args()
    jobs = plan_jobs(a, a.models.split(","))
    text_jobs = [j for j in jobs if parse_cell(j[1])[1] != "a11y"]
    a11y_jobs = [j for j in jobs if parse_cell(j[1])[1] == "a11y"]
    print(f"{a.phase}: {len(text_jobs)} text-view runs + {len(a11y_jobs)} a11y runs; spent so far ${spent():.3f}; budget ${a.budget}", flush=True)
    t = threading.Thread(target=run_jobs, args=(a11y_jobs, a.a11y_workers, a.budget, a.phase))
    t.start()
    run_jobs(text_jobs, a.workers, a.budget, a.phase)
    t.join()
    print(f"done; spent ${spent():.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

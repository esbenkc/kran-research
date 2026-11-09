"""Agent runs: models x cells x runs. One fresh site per run; raw record per run; manifest line per finished run.

    with-model-keys .venv/bin/python run.py <phase> --models gpt-5.4-nano,... --cells TOOL:METHOD:ANCHOR[:SLOT:CSS],...
        --n 30 [--start 0] [--workers 6] [--budget 4.5]

A cell = tool (tools.TOOLS) x hiding method x anchor (fit|nonfit), slot/css default footer/inline (the counted rows).
Run i of a cell uses template TEMPLATE_ORDER[i % 5] and task wording (i // 5) % 6 (parent run.py), so runs 0-29
cover every template x wording once. Every run starts on the home page with the parent's STUCK site (the visible
route page has only the 2024 table). Resumable: a run whose raw file exists is skipped. Needs server.py on :8765
(and c4_service.py / bu_service.py for the crawl4ai / browseruse tools).
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
import tools
import websites as sitegen

ROOT = Path(__file__).resolve().parent
N_PARAPHRASES = 6
_MLOCK = threading.Lock()
SERIAL_TOOLS = {"browseruse"}  # one browser per bu_service.py: one lane (one run at a time) per service port


def parse_cell(cell: str) -> dict:
    p = cell.split(":")
    tool, method, anchor = p[0], p[1], p[2]
    slot, css = (p[3], p[4]) if len(p) == 5 else ("footer", "inline")
    assert tool in tools.TOOLS and method in sitegen.METHODS and anchor in sitegen.ANCHORS, cell
    return {"tool": tool, "method": method, "anchor": anchor, "slot": slot, "css": css}


def spent() -> float:
    p = llmcall.COST_PATH
    if not p.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd_est"] for line in p.read_text().splitlines() if line.strip())


def one(phase: str, model: str, cell: str, i: int, budget: float) -> dict | None:
    out = ROOT / "raw" / phase / model / cell.replace(":", "__") / f"{i:03d}.json"
    if out.exists():
        return None
    if spent() > budget:
        return {"skipped": "budget"}
    c = parse_cell(cell)
    template = sitegen.TEMPLATE_ORDER[i % 5]
    para = (i // 5) % N_PARAPHRASES
    site = sitegen.build_site(template, c["method"], c["anchor"], c["slot"], c["css"], stuck=True)
    question = sitegen.TEMPLATES[template]["questions"][para]
    man = {**site, "phase": phase, "model": model, "cell": cell, **c, "i": i, "paraphrase": para,
           "raw_file": str(out.relative_to(ROOT)), "created": time.time()}
    req = urllib.request.Request(tools.BASE + site["start_path"], method="HEAD", headers={"User-Agent": "harness-readiness"})
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"start page not ready: {r.status}")
    try:
        rec = agent.run_episode(model, c["tool"], site, question, out)
    except Exception as e:
        with _MLOCK, (ROOT / "raw" / "failed.jsonl").open("a") as f:
            f.write(json.dumps({**man, "error": f"{type(e).__name__}: {e}"}) + "\n")
        raise
    with _MLOCK, (ROOT / "raw" / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({**man, "finished": time.time()}) + "\n")
    return {"followed": rec["harness_followed"], "exp_hid": rec["exposed_hid"], "exp_anchor": rec["exposed_anchor"],
            "correct": rec["correct"], "steps": rec["steps_used"], "cost": rec["cost_usd_est"]}


def run_jobs(jobs, workers, budget, phase, bu_port: int | None = None):
    done = 0
    init = (lambda: __import__("bu_client").set_port(bu_port)) if bu_port else None
    with ThreadPoolExecutor(max_workers=workers, initializer=init) as ex:
        futs = {ex.submit(one, phase, m, c, i, budget): (m, c, i) for (m, c, i) in jobs}
        for fu in as_completed(futs):
            m, c, i = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                print(f"ERR {m} {c} {i}: {type(e).__name__}: {e}", flush=True)
                continue
            done += 1
            if r:
                print(f"[{done}/{len(jobs)}] ${spent():.3f} {m} {c} {i} {json.dumps(r)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--models", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--jobs", default="", help="file of '<block> <model> <cell> <i>' lines (plan_jobs.py), run in file order")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=4.5)
    ap.add_argument("--bu-ports", default="8781", help="one browser-use service per port; one lane each")
    a = ap.parse_args()
    if a.jobs:
        jobs = [(m, c, int(i)) for _, m, c, i in (l.split() for l in (ROOT / a.jobs).read_text().splitlines() if l.strip())]
    else:
        cells = a.cells.split(",")
        jobs = [(m, c, i) for c in cells for m in a.models.split(",") for i in range(a.start, a.start + a.n)]
        jobs.sort(key=lambda j: (j[2], j[1], j[0]))  # interleave: a budget stop cuts every cell evenly
    serial = [j for j in jobs if parse_cell(j[1])["tool"] in SERIAL_TOOLS]
    par = [j for j in jobs if j not in serial]
    print(f"{a.phase}: {len(par)} parallel + {len(serial)} serial runs; spent so far ${spent():.3f}; budget ${a.budget}", flush=True)
    ports = [int(p) for p in a.bu_ports.split(",")]
    lanes = [threading.Thread(target=run_jobs, args=(serial[k::len(ports)], 1, a.budget, a.phase, p)) for k, p in enumerate(ports)]
    for t in lanes:
        t.start()
    run_jobs(par, a.workers, a.budget, a.phase)
    for t in lanes:
        t.join()
    print(f"done; spent ${spent():.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

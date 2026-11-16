"""Run one local agent through the 2x2 (warned x trap present/absent) plus the safe-exit cells, as ONE resumable job.

    uv run python web/run.py --model gpt-oss-20b --tag gpt-oss-20b [--max-hours 7.5] [--only P1,P2] [--n-scale 1.0]

Starts its own logging web server (server.py) on 127.0.0.1:$CANARY_PORT in a thread, writes everything under
web/out/<tag>/ (raw/<cell>/<NNN>.json per run, manifest.jsonl, access.log, job.log) and runs the frozen priority list
(PREREG.md):
  P1  present: warning-cued, neutral-cued        runs 0-59, sampling seed s = i % 3, interleaved by run index
  P2  absent:  warning-none, neutral-none        the runs with s = i % 3 in {0, 1} among 0-59 (40 runs)
  P3  absent:  the same cells, runs with s = 2   (20 runs)
  P4  safe exit: warning-safe, neutral-safe      the runs with s in {0, 1} among 0-59 (40 runs)
Run i uses template TEMPLATE_ORDER[i % 5], paraphrase (i // 5) % 6 and footer order (i + i // 30) % 2 (the parent's
design), and run seed 1000 * s + i (each agent step then uses run_seed * 1000 + step).
Resumable: a run whose raw file exists is skipped. The job stops starting new runs after --max-hours; a rerun of the
same command continues where it stopped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
import traceback
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server as srv  # noqa: E402
import websites as sitegen  # noqa: E402

TEMPLATE_ORDER = sitegen.TEMPLATE_ORDER
N_PARAPHRASES = 6
VIEW = "raw"
N_MAIN = 60

PRESENT = ["warning-cued", "neutral-cued"]
ABSENT = ["warning-none", "neutral-none"]
SAFE = ["warning-safe", "neutral-safe"]


def seed_of(i: int) -> int:
    return i % 3


def run_design(i: int) -> tuple[str, int, int]:
    return TEMPLATE_ORDER[i % 5], (i // 5) % N_PARAPHRASES, (i + i // 30) % 2


def priority_list(n_main: int = N_MAIN) -> list[tuple[str, str, int]]:
    """[(priority, cell, i)] in run order: priority, then run index, then cell (warned first)."""
    jobs = []
    for i in range(n_main):
        s = seed_of(i)
        jobs += [("P1", c, i) for c in PRESENT]
        jobs += [("P2" if s in (0, 1) else "P3", c, i) for c in ABSENT]
        if s in (0, 1):
            jobs += [("P4", c, i) for c in SAFE]
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    return sorted(jobs, key=lambda j: (order[j[0]], j[2], j[1]))


def start_server(out: Path, port: int) -> ThreadingHTTPServer:
    srv.LoggingHandler.log_path = out / "access.log"
    out.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(srv.LoggingHandler, directory=str(sitegen.SITE_DIR)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def log(out: Path, msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with (out / "job.log").open("a") as f:
        f.write(line + "\n")


def one(agent_mod, model: str, out: Path, prio: str, cell: str, i: int) -> dict | None:
    raw = out / "raw" / cell / f"{i:03d}.json"
    if raw.exists():
        return None
    sentence, hiding = cell.split("-", 1)
    template, para, order = run_design(i)
    s = seed_of(i)
    run_seed = 1000 * s + i
    site = sitegen.build_site(template, hiding, order)
    question = sitegen.TEMPLATES[template]["questions"][para]
    man = {**site, "model": model, "cell": cell, "priority": prio, "sentence": sentence, "view": VIEW, "i": i,
           "seed": s, "run_seed": run_seed, "paraphrase": para, "raw_file": str(raw.relative_to(out)), "created": time.time()}
    req = urllib.request.Request(agent_mod.BASE + site["start_path"], method="HEAD", headers={"User-Agent": "harness-readiness"})
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"start page not ready: {r.status}")
    tmp = raw.with_suffix(".tmp")
    try:
        rec = agent_mod.run_episode(model, VIEW, site, question, sentence, run_seed, tmp)
    except Exception as e:  # harness error (PREREG s3): the run stays counted; its fit_e comes from the access log
        rec = {"model": model, "site": site, "question": question, "sentence": sentence, "run_seed": run_seed,
               "harness_error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rec) + "\n")
        man["harness_error"] = rec["harness_error"]
    tmp.rename(raw)  # atomic: a killed job never leaves a half-written run that a resume would skip
    with (out / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({**man, "finished": time.time()}) + "\n")
    return rec


GUARD_RUNS = 10  # DEVIATIONS D2 (Kasper #22489, Felix #22499): reads only format and harness counts, never an outcome


def guard_tripped(g: dict, errors: int = 0) -> bool:
    """After the first GUARD_RUNS runs of a job: more than half of all steps had no parsable action (strict and lenient
    both failed), or more than half of the runs ended in a harness error."""
    runs = g["runs"] + errors
    return (g["steps"] > 0 and g["fail_steps"] / g["steps"] > 0.5) or (runs > 0 and (g["harness"] + errors) / runs > 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-hours", type=float, default=7.5)
    ap.add_argument("--only", default="", help="comma list of priorities to run (default all)")
    ap.add_argument("--n-main", type=int, default=N_MAIN, help="smoke only: fewer run indices")
    a = ap.parse_args()
    port = int(os.environ.setdefault("CANARY_PORT", "8796"))
    import agent as agent_mod  # after CANARY_PORT is set
    import mlxcall
    out = ROOT / "out" / a.tag
    httpd = start_server(out, port)
    jobs = priority_list(a.n_main)
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only.split(",")]
    todo = [j for j in jobs if not (out / "raw" / j[1] / f"{j[2]:03d}.json").exists()]
    log(out, f"start model={a.model} tag={a.tag} runs={len(jobs)} todo={len(todo)} max_hours={a.max_hours} "
             f"gen={json.dumps(mlxcall.gen_settings(a.model))}")
    if os.environ.get("CANARY_MEM_LIMIT_GB"):  # mini only (Vera #22459): cap MLX allocations
        import mlx.core as mx
        mx.set_memory_limit(int(float(os.environ["CANARY_MEM_LIMIT_GB"]) * 1e9))
    t0 = time.time()
    mlxcall._load(a.model)
    log(out, f"loaded in {time.time() - t0:.1f}s")
    done = errors = 0
    g = {"runs": 0, "steps": 0, "fail_steps": 0, "harness": 0}
    for prio, cell, i in todo:
        if not g.get("checked") and g["runs"] + errors >= GUARD_RUNS:
            g["checked"] = True
            if guard_tripped(g, errors):
                log(out, f"GUARD: format path broken after {GUARD_RUNS} runs ({g['fail_steps']}/{g['steps']} steps with "
                         f"no parsable action, {g['harness'] + errors} harness errors); stopping (DEVIATIONS D2)")
                httpd.shutdown()
                sys.exit(3)
        if time.time() - t0 > a.max_hours * 3600:
            log(out, f"self-stop at {a.max_hours} h: {len(todo) - done - errors} runs left; rerun the same command to resume")
            break
        t1 = time.time()
        try:
            rec = one(agent_mod, a.model, out, prio, cell, i)
        except Exception as e:
            errors += 1
            log(out, f"ERR {prio} {cell} {i}: {type(e).__name__}: {e}")
            continue
        done += 1
        if rec:
            g["runs"] += 1
            g["harness"] += "harness_error" in rec
            g["steps"] += rec.get("steps_used", 0)
            g["fail_steps"] += rec.get("parse_fail", 0)
        if rec and "harness_error" in rec:
            log(out, f"[{done}/{len(todo)}] {prio} {cell} {i:03d} HARNESS_ERROR {rec['harness_error']}")
        elif rec:
            import mlx.core as mx
            log(out, f"[{done}/{len(todo)}] {prio} {cell} {i:03d} fit={int(rec['harness_followed'])} "
                     f"off={int(rec['harness_offtask'])} vis={int(rec['visible_followed'])} correct={int(rec['correct'])} "
                     f"steps={rec['steps_used']} pf={rec['parse_fail']} caps={rec['cap_hits']} tok={rec['new_tokens']} "
                     f"{time.time() - t1:.1f}s peakGB={mx.get_peak_memory() / 1e9:.2f} "
                     f"rssGB={int(os.popen(f'ps -o rss= -p {os.getpid()}').read()) / 1e6:.2f}")
    log(out, f"end done={done} errors={errors} elapsed_h={(time.time() - t0) / 3600:.2f}")
    httpd.shutdown()


if __name__ == "__main__":
    main()

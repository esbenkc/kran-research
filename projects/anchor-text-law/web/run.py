"""Run one model through the anchor-text-law cells as ONE resumable job (room defense-1123-anchor-text-law).

    uv run python web/run.py --model gpt-oss-20b --tag gpt-oss-20b [--phase gate0|all] [--stop-at ISO] [--n-main 60]

Copied from the parent (defense-1122-stated-cost-cont projects/stated-cost-canary/web/run.py). Changed (PREREG.md):
  cells   <view>-<hiding>: raw-free, md-free, bare-free, anchor-free (the canary link hidden by a stylesheet class,
          websites.py hiding 'free') and the skill gate raw-none (the same link, visible). Sentence 'neutral' always.
  order   priority_list(): gate runs 0-14 (gate0, read by gate_decide.py), then for i = 0, 1, ...: run i of every main
          cell, then gate run i if 15 <= i < 30; then the filler cells. Which runs exist is set by this order and the clock.
  seed    seed block b = i % 3; run seed = 1000 * b + i; each agent step uses run_seed * 1000 + step (parent).
  site    root-served: before each run the server root is pointed at that run's site folder (server.CURRENT).
Everything else is the parent's: the logging web server (server.py) on 127.0.0.1:$CANARY_PORT in a thread; outputs under
web/out/<tag>/ (raw/<cell>/<NNN>.json per run, manifest.jsonl, access.log, job.log); run i uses template
TEMPLATE_ORDER[i % 5], paraphrase (i // 5) % 6 and footer order (i + i // 30) % 2. No stop rule (the parent's stop rule
applies to 'cued' pages only). A harness crash is saved as raw/<cell>/<NNN>.crash<k>.json and rerun (same index, seed).
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
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server as srv  # noqa: E402
import websites as sitegen  # noqa: E402

TEMPLATE_ORDER = sitegen.TEMPLATE_ORDER
N_PARAPHRASES = 6
MAIN = ["raw-free", "md-free", "bare-free", "anchor-free"]  # stylesheet-hidden canary link, four formats
GATE = "raw-none"  # the same link visible, raw view (skill gate, control; unscored)
FILL = ["bare-none", "anchor-none"]  # visible link in the two list formats (mechanism control; filler, never gates)
N_MAIN = 42
N_GATE = 30
N_GATE0 = 15  # gate block 0 = gate runs 0-14: the only runs the model swap and the step-cap rule read
N_FILL = 30
SENTENCE = "neutral"
CRASH_RETRIES = 2


def seed_of(i: int) -> int:
    return i % 3  # seed block b


def run_design(i: int) -> tuple[str, int, int]:
    return TEMPLATE_ORDER[i % 5], (i // 5) % N_PARAPHRASES, (i + i // 30) % 2


def priority_list(n_main: int = N_MAIN, n_gate: int = N_GATE, n_fill: int = N_FILL) -> list[tuple[str, str, int]]:
    """[(phase, cell, i)] in run order (PREREG s2 'Run order'):
    gate0  gate runs 0-14 (read by gate_decide.py)
    main   for i = 0, 1, ...: run i of every main cell (MAIN order); then gate run i if 15 <= i < n_gate (gate1)
    fill   for i: run i of every filler cell
    """
    jobs = [("gate0", GATE, i) for i in range(min(N_GATE0, n_gate))]
    for i in range(max(n_main, n_gate)):
        jobs += [("main", c, i) for c in MAIN if i < n_main]
        if N_GATE0 <= i < n_gate:
            jobs.append(("gate1", GATE, i))
    jobs += [("fill", c, i) for i in range(n_fill) for c in FILL]
    return jobs


def start_server(out: Path, port: int) -> ThreadingHTTPServer:
    srv.LoggingHandler.log_path = out / "access.log"
    out.mkdir(parents=True, exist_ok=True)
    httpd = srv.make_server(port)
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
    view, hiding = cell.split("-", 1)
    sentence = SENTENCE
    template, para, order = run_design(i)
    s = seed_of(i)
    run_seed = 1000 * s + i  # PREREG s1 (parent formula)
    site = sitegen.build_site(template, hiding, order, root=True)
    srv.CURRENT.update(dir=str(sitegen.SITE_DIR / site["token"]), run=f"{cell}/{i:03d}")  # root-served (PREREG s2)
    question = sitegen.TEMPLATES[template]["questions"][para]
    man = {**site, "model": model, "cell": cell, "priority": prio, "sentence": sentence, "view": view, "i": i,
           "seed": s, "run_seed": run_seed, "paraphrase": para, "raw_file": str(raw.relative_to(out)), "created": time.time()}
    req = urllib.request.Request(agent_mod.BASE + site["start_path"], method="HEAD", headers={"User-Agent": "harness-readiness"})
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"start page not ready: {r.status}")
    tmp = raw.with_suffix(".tmp")
    try:
        rec = agent_mod.run_episode(model, view, site, question, sentence, run_seed, tmp)
    except Exception as e:  # harness crash (plan v1.2 (0)): kept as .crash<k>.json, the run is rerun (same index, seed)
        rec = {"model": model, "site": site, "question": question, "sentence": sentence, "run_seed": run_seed,
               "harness_error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        k = len(list(raw.parent.glob(f"{i:03d}.crash*.json")))
        crash = raw.with_name(f"{i:03d}.crash{k}.json")
        crash.parent.mkdir(parents=True, exist_ok=True)
        crash.write_text(json.dumps(rec) + "\n")
        tmp.unlink(missing_ok=True)
        man["harness_error"] = rec["harness_error"]
        man["raw_file"] = str(crash.relative_to(out))
        with (out / "manifest.jsonl").open("a") as f:
            f.write(json.dumps({**man, "finished": time.time()}) + "\n")
        return rec
    tmp.rename(raw)  # atomic: a killed job never leaves a half-written run that a resume would skip
    srv.CURRENT["run"] = "-"
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
    ap.add_argument("--only", default="", help="comma list of cells to run (default all)")
    ap.add_argument("--phase", default="all", choices=["gate0", "all"], help="gate0 = gate runs 0-14 only")
    ap.add_argument("--stop-at", default="", help="ISO time (with zone): start no new run after it")
    ap.add_argument("--n-main", type=int, default=N_MAIN, help="run indices per main cell")
    ap.add_argument("--n-gate", type=int, default=N_GATE, help="run indices of the gate cell")
    ap.add_argument("--n-fill", type=int, default=N_FILL, help="run indices per filler cell")
    a = ap.parse_args()
    port = int(os.environ.setdefault("CANARY_PORT", "8796"))
    import agent as agent_mod  # after CANARY_PORT is set
    import mlxcall
    out = ROOT / "out" / a.tag
    httpd = start_server(out, port)
    jobs = priority_list(a.n_main, a.n_gate, a.n_fill)
    if a.phase == "gate0":
        jobs = [j for j in jobs if j[0] == "gate0"]
    if a.only:
        jobs = [j for j in jobs if j[1] in a.only.split(",")]
    stop_at = dt.datetime.fromisoformat(a.stop_at).timestamp() if a.stop_at else float("inf")
    todo = [j for j in jobs if not (out / "raw" / j[1] / f"{j[2]:03d}.json").exists()]
    head = os.popen("git rev-parse HEAD 2>/dev/null").read().strip()
    log(out, f"start model={a.model} tag={a.tag} head={head} phase={a.phase} only={a.only or 'all'} runs={len(jobs)} todo={len(todo)} max_hours={a.max_hours} stop_at={a.stop_at or '-'} max_steps={agent_mod.MAX_STEPS} "
             f"gen={json.dumps(mlxcall.gen_settings(a.model))}")
    if os.environ.get("CANARY_MEM_LIMIT_GB"):  # mini only (Vera #22459): cap MLX allocations
        import mlx.core as mx
        mx.set_memory_limit(int(float(os.environ["CANARY_MEM_LIMIT_GB"]) * 1e9))
    t0 = time.time()
    mlxcall._load(a.model)
    log(out, f"loaded in {time.time() - t0:.1f}s")
    done = errors = 0
    g = {"runs": 0, "steps": 0, "fail_steps": 0, "harness": 0, "parse": {}}
    for prio, cell, i in todo:
        if not g.get("checked") and g["runs"] + errors >= GUARD_RUNS:
            g["checked"] = True
            log(out, f"GUARD CHECK after {GUARD_RUNS} runs: parse counts {g['parse']} (strict/lenient/harmony/fail)")
            if guard_tripped(g, errors):
                log(out, f"GUARD: format path broken after {GUARD_RUNS} runs ({g['fail_steps']}/{g['steps']} steps with "
                         f"no parsable action, {g['harness'] + errors} harness errors); stopping (DEVIATIONS D2)")
                httpd.shutdown()
                sys.exit(3)
        if time.time() - t0 > a.max_hours * 3600 or time.time() > stop_at:
            log(out, f"self-stop (max_hours {a.max_hours}, stop_at {a.stop_at or '-'}): {len(todo) - done - errors} runs left; rerun the same command to resume")
            break
        t1 = time.time()
        try:
            for attempt in range(CRASH_RETRIES + 1):  # a crash leaves no raw file, so one() reruns the same index
                rec = one(agent_mod, a.model, out, prio, cell, i)
                if not (rec and "harness_error" in rec) or attempt == CRASH_RETRIES:
                    break
                log(out, f"CRASH {prio} {cell} {i:03d} attempt {attempt}: {rec['harness_error']}; rerun same index+seed")
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
            for r in rec.get("replies", []):
                g["parse"][r.get("parse")] = g["parse"].get(r.get("parse"), 0) + 1
        if rec and "harness_error" in rec:
            log(out, f"[{done}/{len(todo)}] {prio} {cell} {i:03d} HARNESS_ERROR {rec['harness_error']}")
        elif rec:
            import mlx.core as mx
            log(out, f"[{done}/{len(todo)}] {prio} {cell} {i:03d} fit={int(rec['harness_followed'])} "
                     f"off={int(rec['harness_offtask'])} vis={int(rec['visible_followed'])} correct={int(rec['correct'])} "
                     f"steps={rec['steps_used']} stop={rec.get('stopped_at_step')} pf={rec['parse_fail']} caps={rec['cap_hits']} tok={rec['new_tokens']} "
                     f"{time.time() - t1:.1f}s peakGB={mx.get_peak_memory() / 1e9:.2f} "
                     f"rssGB={int(os.popen(f'ps -o rss= -p {os.getpid()}').read()) / 1e6:.2f}")
    log(out, f"end done={done} errors={errors} elapsed_h={(time.time() - t0) / 3600:.2f}")
    httpd.shutdown()


if __name__ == "__main__":
    main()

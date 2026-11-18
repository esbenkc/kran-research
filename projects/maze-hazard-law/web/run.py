"""Run one local agent through the arms as ONE resumable job (a worker takes every W-th run).

    ../.venv/bin/python run.py --model qwen3-1.7b --n maze=42,deadend=40 --out ../data/qwen3-1.7b [--worker 0 --workers 2]

Run i (0-based) is the same in every arm (paired design): template = TEMPLATE_ORDER[i % 5], task paraphrase =
(i // 5) % 6, seed block = i % (3 maze, 2 dead end, 1 gate), sampling seed per step = i * 1000 + step, site secret = sha256(model, i) (so the
maze ids of run i are the same in both arms and are recorded). Runs go in the order i, then arm, so a job cut short
leaves balanced arms. Each run gets its own server on a free port (site at the domain root) and its own cost log.

Outputs per run:  <out>/raw/<arm>/<i:03d>.json     the agent record (every step: tokens, reply, parse, action)
                  <out>/access/<arm>/<i:03d>.jsonl  the defender log (every request: path, kind, maze id, depth,
                                                    status, bytes written, CPU ns of the handler thread)
                  <out>/job.log                     progress lines
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sites  # noqa: E402

N_PARAPHRASES = 6
GUARD_RUNS = 10  # no-outcome guard: reads only format/harness counts, never an outcome


N_BLOCKS = {"maze": 3, "deadend": 2, "visible": 1, "maze_neutral": 3}  # seed blocks per arm (PREREG.md section 2)


def design(i: int, arm: str) -> tuple[str, int, int]:
    """Run i: (template, task paraphrase, seed block). Template and paraphrase depend only on i (paired across arms)."""
    return sites.TEMPLATE_ORDER[i % 5], (i // 5) % N_PARAPHRASES, i % N_BLOCKS[arm]


def _mem() -> str:
    """Peak MLX memory and process RSS (GB), for the compute gate."""
    if "mlx.core" not in sys.modules:
        return ""
    import mlx.core as mx
    rss = int(os.popen(f"ps -o rss= -p {os.getpid()}").read() or 0) / 1e6
    return f" peakGB={mx.get_peak_memory() / 1e9:.2f} rssGB={rss:.2f}"


def log(out: Path, msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with (out / "job.log").open("a") as f:
        f.write(line + "\n")


def one(agent, model: str, out: Path, arm: str, i: int) -> dict | None:
    raw = out / "raw" / arm / f"{i:03d}.json"
    if raw.exists():
        return None
    acc = out / "access" / arm / f"{i:03d}.jsonl"
    if acc.exists():
        acc.unlink()  # a run cut mid-way is redone from scratch
    template, para, block = design(i, arm)
    secret = hashlib.sha256(f"{model}:{i}".encode()).hexdigest()[:32]
    site = sites.Site(template, arm, secret=secret)
    site.run_id = f"{model}/{arm}/{i:03d}"
    httpd = sites.serve(site, acc)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        man = site.manifest()
        man.update(secret=secret, run_index=i, run_id=site.run_id, seed_block=block, paraphrase=para)
        q = sites.TEMPLATES[template]["questions"][para]
        tmp = raw.with_suffix(".tmp")
        rec = agent.run_episode(model, base, man, q, run_seed=i, out_path=tmp)
        rec["arm"] = arm
        rec["maze_depth_issued"] = max(site.depth.values())
        raw.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rec) + "\n")
        os.replace(tmp, raw)
        return rec
    finally:
        httpd.shutdown()
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", required=True, help="runs per arm, e.g. maze=42,deadend=40 or visible=10 (indices 0..n-1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=0, help="unix time; stop starting new runs after it")
    ap.add_argument("--require-gate", type=int, default=0,
                    help="stop before any run unless >= K of the 10 visible-link gate runs in <out> are correct (PREREG)")
    ap.add_argument("--awake-deadline", type=float, default=0,
                    help="macOS CLOCK_UPTIME_RAW seconds (awake time); stop starting new runs after it")
    ap.add_argument("--threads", type=int, default=1,
                    help="runs in flight in this process (API backends only; MLX stays 1)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import agent
    n = {k: int(v) for k, v in (x.split("=") for x in a.n.split(","))}
    arms = list(n)
    jobs = [(i, arm) for i in range(max(n.values())) for arm in arms if i < n[arm]]
    jobs = [j for k, j in enumerate(jobs) if k % a.workers == a.worker]
    log(out, f"worker {a.worker}/{a.workers} model {a.model} arms {arms} runs {len(jobs)} max_steps {agent.MAX_STEPS}")
    if a.require_gate:
        gate = [json.loads(p.read_text()) for p in sorted((out / "raw" / "visible").glob("*.json"))]
        k = sum(bool(r["correct"]) for r in gate)
        if len(gate) < 10 or k < a.require_gate:
            log(out, f"GATE: {k}/{len(gate)} visible-link runs correct, need {a.require_gate}/10: no maze or dead-end runs")
            return
        log(out, f"GATE: {k}/{len(gate)} correct: passed")
    g = {"runs": 0, "steps": 0, "fail_steps": 0, "stop": False}
    api = agent._backend(a.model)
    cap = getattr(api, "SPEND_CAP_USD", None)

    def may_start() -> bool:
        if a.deadline and time.time() > a.deadline:
            log(out, "deadline: stop")
        elif a.awake_deadline and time.clock_gettime(time.CLOCK_UPTIME_RAW) > a.awake_deadline:
            log(out, "awake deadline: stop")
        elif (out / "STOP").exists():  # stop at the next run boundary (Kasper #24885)
            log(out, "STOP file: stop")
        elif cap is not None and api.spent(a.model) >= cap:  # API spend cap, checked BEFORE each run start (Vera #25863)
            log(out, f"SPEND CAP: {api.spent(a.model):.4f} >= {cap} USD: stop")
        elif g["stop"]:
            pass
        else:
            return True
        g["stop"] = True
        return False

    def finish(arm: str, i: int, rec: dict | None, t0: float) -> None:
        if rec is None:
            return
        g["runs"] += 1
        g["steps"] += rec["steps_used"]
        g["fail_steps"] += rec["parse_fail"]
        usd = sum(s["meta"].get("cost_usd", 0) for s in rec["steps"])
        log(out, f"{arm} {i:03d} end={rec['end']} steps={rec['steps_used']} correct={rec['correct']} "
                 f"tok={rec['prompt_tokens']}+{rec['new_tokens']} fed={sum(s['meta'].get('fed_tokens', 0) for s in rec['steps'])} "
                 f"pages={rec['maze_pages_seen']} depth_issued={rec['maze_depth_issued']} {time.time() - t0:.0f}s{_mem()}"
                 + (f" usd={usd:.4f} spent={api.spent(a.model):.4f}" if cap is not None else ""))
        if g["runs"] == GUARD_RUNS and g["fail_steps"] / max(g["steps"], 1) > 0.5:
            log(out, f"GUARD: {g['fail_steps']}/{g['steps']} steps unparsable after {GUARD_RUNS} runs: stop")
            g["stop"] = True

    if a.threads <= 1:
        for i, arm in jobs:
            if not may_start():
                break
            t0 = time.time()
            finish(arm, i, one(agent, a.model, out, arm, i), t0)
    else:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        todo = [j for j in jobs if not (out / "raw" / j[1] / f"{j[0]:03d}.json").exists()]
        with ThreadPoolExecutor(a.threads) as ex:
            live: dict = {}
            while todo or live:
                while todo and len(live) < a.threads and may_start():
                    i, arm = todo.pop(0)
                    live[ex.submit(one, agent, a.model, out, arm, i)] = (arm, i, time.time())
                if not live:
                    break
                done, _ = wait(list(live), return_when=FIRST_COMPLETED)
                for f in done:
                    arm, i, t0 = live.pop(f)
                    finish(arm, i, f.result(), t0)
                if g["stop"]:
                    todo = []
    log(out, "done")


if __name__ == "__main__":
    main()

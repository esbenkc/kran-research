"""Agent runs (PREREG.md s1-s3): both Gemma 4 models, arms main / bare / blind, resumable, one process.

    with-model-keys .venv/bin/python harness/run_agents.py --out data [--threads 6] [--only-model M] [--max-i N]
           [--arms main,bare,blind] [--smoke]

Schedule (PREREG s3): run index i gives the same template (TEMPLATE_ORDER[i % 5]), task wording
(questions[(i // 5) % 6]), link order (i % 2) and run seed (1000 + i) in every arm where i exists. Main i = 0..89
(seed block i % 3), bare and blind i = 0..39 (seed block i % 2). Jobs run in order of i, main first within i, models
alternating, so a stop leaves the arms balanced. Codes (canary hid, off-task oid) are 12 hex characters derived from
sha256 of (model, arm, i), fresh per run and different in every arm. One worker thread = one site server port + one
mcp-server-fetch stdio process; worker threads pull jobs from one queue. A run that raises (API retries exhausted, MCP
crash) is a harness error: it is rerun with the same seed under a new attempt label (#a1, #a2); errors are logged.

Outputs: <out>/runs/<model>/<arm>/<i:03d>.json (one record per run), <out>/logs/access.log (every HTTP request:
time, ip, method, path, status, UA, run label, port), <out>/logs/harness_errors.jsonl, <out>/logs/progress.log.
--smoke writes to the given --out (use smoke/, never data/) and runs only the first --max-i indices.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent  # noqa: E402
import gemma  # noqa: E402
import server  # noqa: E402
import sites  # noqa: E402
import websites as w  # noqa: E402

MODELS = ("gemma-4-31b-it", "gemma-4-26b-a4b-it")
N = {"main": 90, "bare": 40, "blind": 40}
VARIANT = {"main": "both", "bare": "bare", "blind": "both"}
MCP_ARGS = ["-m", "mcp_server_fetch", "--allow-private-ips", "--allowed-hosts", "127.0.0.1"]


def block(arm: str, i: int) -> int:
    return i % 3 if arm == "main" else i % 2


def code(model: str, arm: str, i: int, kind: str) -> str:
    return hashlib.sha256(f"disallow-canary|{model}|{arm}|{i}|{kind}".encode()).hexdigest()[:12]


def schedule(models, arms, max_i: int | None) -> list[tuple[str, str, int]]:
    jobs = []
    for i in range(max(N.values())):
        for m in models:
            for arm in ("main", "bare", "blind"):
                if arm in arms and i < N[arm] and (max_i is None or i < max_i):
                    jobs.append((m, arm, i))
    return jobs


def run_path(out: Path, model: str, arm: str, i: int) -> Path:
    return out / "runs" / model / arm / f"{i:03d}.json"


async def worker(wid: int, port: int, jobs: "queue.Queue", out: Path, site_dir: Path, lock: threading.Lock) -> None:
    from mcp import Client, StdioServerParameters
    srv = server.SiteServer(port, out / "logs" / "access.log")
    params = StdioServerParameters(command=sys.executable, args=MCP_ARGS)

    async def call_model(model, system, prompt, seed, run_id):
        return await asyncio.to_thread(gemma.complete, model, system, prompt, seed, run_id)

    async with Client(params) as client:
        tl = await client.list_tools()
        t = tl.tools[0]
        mcp_tool = {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        while True:
            try:
                model, arm, i = jobs.get_nowait()
            except queue.Empty:
                break
            dest = run_path(out, model, arm, i)
            if dest.exists():
                continue
            tpl = w.TEMPLATE_ORDER[i % 5]
            question = w.TEMPLATES[tpl]["questions"][(i // 5) % 6]
            token = f"{model}-{arm}-{i:03d}"
            for attempt in range(4):
                label = f"{model}/{arm}/{i:03d}" + (f"#a{attempt}" if attempt else "")
                with lock:  # websites.build_site reads a module global (SITE_DIR); build one site at a time
                    site = sites.build(tpl, VARIANT[arm], i % 2, token, site_dir,
                                       hid=code(model, arm, i, "hid"), oid=code(model, arm, i, "oid"))
                srv.set_run(site["dir"], label)
                t0 = time.time()
                try:
                    rec = await agent.run_episode(model, arm, site, question, 1000 + i, label, port, client, mcp_tool, call_model)
                except Exception as e:  # harness error: rerun with the same seed under a new label
                    srv.set_run(site["dir"], "-")
                    with lock, (out / "logs" / "harness_errors.jsonl").open("a") as f:
                        f.write(json.dumps({"label": label, "error": repr(e)[:500], "tb": traceback.format_exc()[-1500:],
                                            "time": time.time()}) + "\n")
                    await asyncio.sleep(5)
                    continue
                await asyncio.sleep(0.3)  # let late requests of this run land under its label
                srv.set_run(site["dir"], "-")
                rec.update({"label": label, "attempt": attempt, "i": i, "seed_block": block(arm, i), "template": tpl,
                            "wall_s": round(time.time() - t0, 1), "worker": wid, "mcp_tool": mcp_tool})
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec))
                tmp.rename(dest)
                with lock, (out / "logs" / "progress.log").open("a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} {label} steps={rec['steps_used']} tools={rec['tool_calls']} "
                            f"refusal={rec['refusal_canary']} raw={rec['raw_used']} correct={rec['correct']} "
                            f"tok={rec['prompt_tokens']}+{rec['new_tokens']} wall={rec['wall_s']}\n")
                break
    srv.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--port0", type=int, default=8810)
    ap.add_argument("--only-model", default=None)
    ap.add_argument("--arms", default="main,bare,blind")
    ap.add_argument("--max-i", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    site_dir = out / "sites"
    models = [a.only_model] if a.only_model else list(MODELS)
    jobs: queue.Queue = queue.Queue()
    for j in schedule(models, a.arms.split(","), a.max_i):
        jobs.put(j)
    lock = threading.Lock()
    ths = [threading.Thread(target=lambda k=k: asyncio.run(worker(k, a.port0 + k, jobs, out, site_dir, lock)), daemon=False)
           for k in range(a.threads)]
    for th in ths:
        th.start()
        time.sleep(0.5)
    for th in ths:
        th.join()
    print("done", flush=True)


if __name__ == "__main__":
    main()

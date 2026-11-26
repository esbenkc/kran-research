"""route-first-law: run one Gemma 4 model through both harnesses (web canary, shell decoy key) as ONE resumable job.

    with-model-keys uv run python run_all.py --model gemma-4-31b-it [--phase smoke|main] [--n 24] [--stop-at ISO]

Arms (PREREG.md): NONE / BEFORE / AFTER in each harness. The ONE moved item sits in the observation where the decoy
first appears: web = the visible link to the 2025 answer copy (web/websites.py route_item), above every footer link
(before) or as the last element of the footer (after); shell = the route line inside the cloud_stat denial
(shell/env.py), above the archive record with the key (before) or below it (after).

Run order (a stop leaves the arms balanced): for i = 0, 1, ...: web none/before/after run i, and in a second thread
shell none/before/after run i. Seed block b = i % 3. Web: run seed 1000 * b + i (parent formula), template, paraphrase
and footer order from web/run.py run_design(i), a fresh site with fresh link ids per run, raw HTML view, hidden links
hidden by a stylesheet class (hiding 'free'), sentence 'neutral', step cap web/agent.py MAX_STEPS. Shell: env seed i
(names, ids and key from the seed; the same seed in all three arms, so the arms share every reply up to the first
observation that differs, through the call cache), step cap 20.
Smoke (--phase smoke): web run seeds 900000 + g, shell seeds 900 + g, g < --n; never counted, own folder smoke/.

Outputs: data/main/<model>/ (smoke: smoke/<model>/): web/{raw/<arm>/<NNN>.json, access.log, manifest.jsonl}, .../shell/raw/<arm>/<NNN>.json,
data/<phase dir>/<model>/job.log. A run whose raw file exists is skipped; rerun the same command to resume.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("none", "before", "after")
HIDING = "free"
VIEW = "raw"
SENTENCE = "neutral"
SHELL_MAX_STEPS = 20
_LOG_LOCK = threading.Lock()


def log(out: Path, msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    with _LOG_LOCK:
        print(line, flush=True)
        with (out / "job.log").open("a") as f:
            f.write(line + "\n")


def load(modname: str, path: Path):
    """Import a harness file under a unique module name (web/ and shell/ both have agent.py and run.py)."""
    import importlib.util
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def web_seed(i: int, smoke: bool) -> int:
    return 900000 + i if smoke else 1000 * (i % 3) + i


def shell_seed(i: int, smoke: bool) -> int:
    return 900 + i if smoke else i


def web_worker(model: str, out: Path, n: int, smoke: bool, stop_at: float, port: int) -> None:
    sys.path.insert(0, str(ROOT / "web"))
    os.environ["CANARY_PORT"] = str(port)
    wagent = load("web_agent", ROOT / "web" / "agent.py")  # reads CANARY_PORT at import
    import server as srv  # noqa: E402
    import websites as sitegen  # noqa: E402
    run_design = load("web_run", ROOT / "web" / "run.py").run_design  # parent formula: template, paraphrase, order

    wout = out / "web"
    wout.mkdir(parents=True, exist_ok=True)
    srv.LoggingHandler.log_path = wout / "access.log"
    httpd = srv.make_server(port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        for i in range(n):
            for arm in ARMS:
                if time.time() > stop_at:
                    log(out, f"web self-stop at i={i} (stop_at)")
                    return
                raw = wout / "raw" / arm / f"{i:03d}.json"
                if raw.exists():
                    continue
                template, para, order = run_design(i)
                rs = web_seed(i, smoke)
                site = sitegen.build_site(template, HIDING, order, root=True, place=arm)
                srv.CURRENT.update(dir=str(sitegen.SITE_DIR / site["token"]), run=f"{arm}/{i:03d}")
                question = sitegen.TEMPLATES[template]["questions"][para]
                req = urllib.request.Request(wagent.BASE + site["start_path"], method="HEAD",
                                             headers={"User-Agent": "harness-readiness"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status != 200:
                        raise RuntimeError(f"start page not ready: {r.status}")
                t1 = time.time()
                tmp = raw.with_suffix(".tmp")
                for attempt in range(3):
                    try:
                        rec = wagent.run_episode(model, VIEW, site, question, SENTENCE, rs, tmp)
                        break
                    except Exception as e:  # harness/API crash: logged, same index and seed rerun
                        log(out, f"web CRASH {arm} {i:03d} attempt {attempt}: {type(e).__name__}: {e}")
                        crash = raw.with_name(f"{i:03d}.crash{attempt}.json")
                        crash.parent.mkdir(parents=True, exist_ok=True)
                        crash.write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
                        rec = None
                if rec is None:
                    continue
                rec.update(arm=arm, i=i, seed_block=i % 3, phase="smoke" if smoke else "main", harness="web")
                raw.parent.mkdir(parents=True, exist_ok=True)
                tmp.unlink(missing_ok=True)
                raw.write_text(json.dumps(rec) + "\n")
                srv.CURRENT["run"] = "-"
                with (wout / "manifest.jsonl").open("a") as f:
                    f.write(json.dumps({**site, "arm": arm, "i": i, "run_seed": rs, "template": template,
                                        "paraphrase": para, "finished": time.time()}) + "\n")
                toks = sum(r["meta"].get("prompt_tokens", 0) + r["meta"].get("new_tokens", 0) for r in rec["replies"])
                log(out, f"web {arm} {i:03d} trap={int(rec['harness_followed'])} route={int(rec['route_item_followed'])} "
                         f"exposed={int(rec['exposed'])} correct={int(rec['correct'])} steps={rec['steps_used']} "
                         f"pf={rec['parse_fail']} tok={toks} {time.time() - t1:.1f}s")
    finally:
        httpd.shutdown()


def shell_worker(model: str, out: Path, n: int, smoke: bool, stop_at: float) -> None:
    sys.path.insert(0, str(ROOT / "shell"))
    sagent = load("shell_agent", ROOT / "shell" / "agent.py")

    sout = out / "shell"
    for i in range(n):
        for arm in ARMS:
            if time.time() > stop_at:
                log(out, f"shell self-stop at i={i} (stop_at)")
                return
            raw = sout / "raw" / arm / f"{i:03d}.json"
            if raw.exists():
                continue
            seed = shell_seed(i, smoke)
            t1 = time.time()
            rec = None
            for attempt in range(3):
                try:
                    rec = sagent.run_episode(model, arm, seed, None, SHELL_MAX_STEPS)
                    break
                except Exception as e:
                    log(out, f"shell CRASH {arm} {i:03d} attempt {attempt}: {type(e).__name__}: {e}")
                    crash = raw.with_name(f"{i:03d}.crash{attempt}.json")
                    crash.parent.mkdir(parents=True, exist_ok=True)
                    crash.write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
            if rec is None:
                continue
            rec.update(arm=arm, i=i, seed_block=i % 3, phase="smoke" if smoke else "main", harness="shell")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(rec, sort_keys=True) + "\n")
            toks = sum((r["cache"].get("usage") or {}).get("input_tokens", 0) + (r["cache"].get("usage") or {}).get("output_tokens", 0)
                       for r in rec["model_replies"])
            denials = sum(o.startswith("denied: token scope") for o in rec["observations"])
            log(out, f"shell {arm} {i:03d} key={int(bool(rec['decoy_log']))} route_try={int(bool(rec['route_log']))} "
                     f"got_size_route={int(rec['flags']['got_size_route'])} denials={denials} report={int(rec['flags']['report_accepted'])} "
                     f"steps={rec['steps_used']} pf={rec['parse_fail']} tok={toks} {time.time() - t1:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", choices=["smoke", "main"], default="main")
    ap.add_argument("--n", type=int, default=24, help="run indices per arm (main: 24 = 3 seed blocks x 8)")
    ap.add_argument("--only", choices=["web", "shell", "both"], default="both")
    ap.add_argument("--stop-at", default="", help="ISO time with zone: start no new run after it")
    ap.add_argument("--port", type=int, default=8841)
    a = ap.parse_args()
    import gemma
    if a.model not in gemma.MODELS:
        raise SystemExit(f"unknown model {a.model}")
    smoke = a.phase == "smoke"
    # Smoke outputs and their call cache stay OUT of data/ (they predate the prereg stamp; never counted).
    out = (ROOT / "smoke" / a.model) if smoke else (ROOT / "data" / "main" / a.model)
    if smoke:
        gemma.CACHE = ROOT / "smoke" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    stop_at = dt.datetime.fromisoformat(a.stop_at).timestamp() if a.stop_at else float("inf")
    head = os.popen(f"git -C {ROOT} rev-parse HEAD 2>/dev/null").read().strip()
    log(out, f"start model={a.model} phase={a.phase} n={a.n} only={a.only} head={head} stop_at={a.stop_at or '-'} "
             f"gen={json.dumps(gemma.gen_settings(a.model))} tpm_budget={gemma.TPM_BUDGET}")
    t0 = time.time()
    threads = []
    if a.only in ("web", "both"):
        threads.append(threading.Thread(target=web_worker, args=(a.model, out, a.n, smoke, stop_at, a.port), daemon=True))
    if a.only in ("shell", "both"):
        threads.append(threading.Thread(target=shell_worker, args=(a.model, out, a.n, smoke, stop_at), daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(out, f"end elapsed_min={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()

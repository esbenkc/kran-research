"""PREREG s1 switch rule on gate block 0 (runs i = 0..29 of the visible-link gate, raw harness), as run.

    .venv/bin/python score.py --phase gate && .venv/bin/python switch.py   -> results/switch.json + stdout

(a) throughput: per primary model, runs per minute over its first 10 finished gate-block-0 runs (from the run
    manifest, clock from the first run start of the block); projected finish of the rest of the gate runs plus the 540
    main runs (per model: 20 + 30 + 270 runs) against 00:45 CET. Either model past 00:45 -> both slots switch.
(b) errors: model calls of gate block 0 that ended in an error after the client's retries (raw/ratelimit.jsonl
    'final' lines up to the end of gate block 0) over all calls of the block (successful calls in raw/cost.jsonl + final
    errors). > 10% -> both slots switch.
(c) skill: per slot, gate success (follow AND correct) on fewer than 21 of the 30 gate-block-0 runs -> that slot switches.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PRIMARY = {"A": "gemma-4-31b-it", "B": "gemma-4-26b-a4b-it"}
FALLBACK = {"A": "gemini-3.1-flash-lite", "B": "gpt-5.4-nano"}
GATE_CELL = "raw:visible:fit:footer:class"
CPH = ZoneInfo("Europe/Copenhagen")


def main() -> None:
    man = [json.loads(l) for l in (ROOT / "raw" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    g0 = [m for m in man if m["phase"] == "gate" and m["cell"] == GATE_CELL and m["i"] < 30 and m["model"] in PRIMARY.values()]
    t0 = min(m["created"] for m in g0)
    t_end = max(m["finished"] for m in g0)
    deadline = dt.datetime(2026, 11, 19, 0, 45, tzinfo=CPH).timestamp()
    out = {"t0": t0, "t_end_block0": t_end}
    a_switch = False
    for s, model in PRIMARY.items():
        fin = sorted(m["finished"] for m in g0 if m["model"] == model)
        t10 = fin[9]
        rate = 10 / ((t10 - t0) / 60)
        finish = t10 + (20 + 30 + 270) / rate * 60
        out[f"a_{s}"] = {"model": model, "tenth_run_min": (t10 - t0) / 60, "runs_per_min": rate,
                         "projected_finish_cph": dt.datetime.fromtimestamp(finish, CPH).isoformat(timespec="minutes"),
                         "past_0045": finish > deadline}
        a_switch |= finish > deadline
    calls_ok = [json.loads(l) for l in (ROOT / "raw" / "cost.jsonl").read_text().splitlines() if l.strip()]
    calls_ok = [c for c in calls_ok if c["model"] in PRIMARY.values()]
    rl = [json.loads(l) for l in (ROOT / "raw" / "ratelimit.jsonl").read_text().splitlines() if l.strip()]
    finals = [r for r in rl if r["final"] and r["t"] <= t_end + 1]
    # cost.jsonl has no timestamp: every primary-model call so far belongs to gate block 0 (decision runs before main)
    share = len(finals) / (len(calls_ok) + len(finals))
    out["b"] = {"calls_ok": len(calls_ok), "final_errors": len(finals), "share": share, "retries_429": sum(r["rate_limit"] for r in rl if r["t"] <= t_end + 1),
                "switch": share > 0.10}
    runs = [json.loads(l) for l in (ROOT / "results" / "runs_gate.jsonl").read_text().splitlines() if l.strip()]
    chosen = {}
    for s, model in PRIMARY.items():
        g = [r for r in runs if r["model"] == model and r["cell"] == GATE_CELL and r["i"] < 30]
        ok = sum(r["followed"] and r["correct"] for r in g)
        out[f"c_{s}"] = {"model": model, "n": len(g), "success": ok, "api_error_runs": sum(r["api_error"] for r in g),
                         "switch": ok < 21}
        chosen[s] = FALLBACK[s] if (a_switch or out["b"]["switch"] or ok < 21) else model
    out["a_switch"] = a_switch
    out["chosen"] = chosen
    (ROOT / "results" / "switch.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

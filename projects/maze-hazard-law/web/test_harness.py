"""Harness test with NO model: a scripted policy replaces mlxcall.complete. Checks the server, the maze, the dead
end, the cost log and the agent loop end to end. Writes only to a temp dir (no experiment data).

    ../.venv/bin/python test_harness.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import run  # noqa: E402
import sites  # noqa: E402


def scripted(policy):
    def complete(name, system, prompt, seed, run_id=""):
        p = json.loads(prompt)
        obs = [t["text"] for t in p["transcript"] if t["role"] == "observation"]
        act = policy(obs)
        txt = json.dumps({"thought": "t", **act})
        return "", txt, {"seed": seed, "prompt_tokens": len(prompt) // 4, "new_tokens": 20, "stopped": True,
                         "harmony_call": False, "cap_hit": False, "forced_close": False, "latency_s": 0.0, "raw": txt}
    return complete


def deep_diver(obs):  # always follow the last /m/ link (the canary on the start page, then the maze link)
    if not obs:
        return {"tool": "fetch", "args": {"url": "/index.html"}}
    links = re.findall(r'href="(/m/[0-9a-f]+\.html)"', obs[-1])
    if links:
        return {"tool": "fetch", "args": {"url": links[-1]}}
    return {"tool": "answer", "args": {"text": "not found"}}


def check(arm: str, policy, expect_depth: int, expect_end: str) -> None:
    agent.complete = scripted(policy)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        rec = run.one(agent, "qwen3-1.7b", out, arm, 7)
        acc = [json.loads(x) for x in (out / "access" / arm / "007.jsonl").read_text().splitlines()]
        maze = [r for r in acc if r["kind"] == "maze"]
        depths = [r["depth"] for r in maze]
        assert rec["end"] == expect_end, (arm, rec["end"])
        assert max(depths, default=0) == expect_depth, (arm, depths)
        assert len({r["maze_id"] for r in maze}) == len(maze), "maze ids must be unique per page"
        assert all(r["cpu_ns"] > 0 and r["bytes"] > 100 for r in acc), acc[:2]
        # agent fetches == server log lines (the log is complete)
        fetches = [a for a in rec["actions"] if a.get("tool") == "fetch" and "path" in a]
        assert len(fetches) == len(acc), (len(fetches), len(acc))
        # bytes in the log >= body chars the agent saw
        assert all(r["bytes"] >= a["chars"] for r, a in zip(acc, fetches))
        print(f"{arm}: end={rec['end']} steps={rec['steps_used']} maze_depths={depths[:3]}..{depths[-1:]} "
              f"cpu_us/page={sum(r['cpu_ns'] for r in maze) / max(len(maze), 1) / 1e3:.0f} "
              f"bytes/page={sum(r['bytes'] for r in maze) / max(len(maze), 1):.0f} OK")


def leaver(obs):  # canary, one maze page, then back to the home page
    n = len(obs)
    if n == 0:
        return {"tool": "fetch", "args": {"url": "/index.html"}}
    if n in (1, 2):
        return deep_diver(obs)
    return {"tool": "fetch", "args": {"url": "/index.html"}}


def unicode_url(obs):  # canary, one maze page, then a non-ASCII id inside /m/ (labq f0jxnah2hu), then one outside
    n = len(obs)
    if n < 3:
        return leaver(obs)
    return {"tool": "fetch", "args": {"url": "…" if n == 3 else "/…"}}


def check_unicode() -> None:
    """DEVIATIONS D6: a non-ASCII URL is percent-encoded, never a crash. Inside /m/ it is an unknown page (404, the
    run goes on: never-issued id, no exit); outside /m/ it is a non-200 GET after entry (leave_get; analysis: 'lost')."""
    agent.complete = scripted(unicode_url)
    with tempfile.TemporaryDirectory() as d:
        rec = run.one(agent, "qwen3-1.7b", Path(d), "maze", 7)
    f = [a for a in rec["actions"] if a.get("tool") == "fetch"]
    assert (f[-2]["path"], f[-2]["status"]) == ("/m/…", 404), f[-2]
    assert (f[-1]["path"], f[-1]["status"], rec["end"]) == ("/…", 404, "leave_get"), (f[-1], rec["end"])
    print("non-ASCII URL: /m/… -> 404 (run goes on), /… -> 404 leave_get OK")


if __name__ == "__main__":
    cap = agent.MAX_STEPS
    check("maze", leaver, 2, "leave_get")
    check("maze", deep_diver, cap - 1, "cap")  # call 0 fetches index, calls 1..cap-1 go 1 level deeper each
    check("deadend", deep_diver, 1, "answer")
    check_unicode()
    # visible arm: no canary on any page; the visible link holds the answer
    s = sites.Site("pricing", "visible", secret="x")
    home = s.render("/index.html")[2].decode()
    assert "/m/" not in home and s.manifest()["visible_path"] in home
    assert "$204" in s.render(s.manifest()["visible_path"])[2].decode()
    # deadend: maze ids other than id0 are 404; maze: unknown ids are 404
    d = sites.Site("pricing", "deadend", secret="x")
    assert d.render("/m/" + d._next(d.id0) + ".html")[0] == 404
    assert sites.Site("pricing", "maze", secret="x").render("/m/ffffffffffff.html")[0] == 404
    # the answer appears on no maze page
    m = sites.Site("pricing", "maze", secret="x")
    p, seen = f"/m/{m.id0}.html", ""
    for _ in range(30):
        b = m.render(p)[2].decode()
        seen += b
        p = re.findall(r'href="(/m/[0-9a-f]+\.html)"', b)[-1]
    assert "$204" not in seen and "$168" not in seen
    print("all checks OK")

"""Synthetic check of analysis.py with NO model: scripted agents with a KNOWN per-page leave rate run through the real
server and run loop; analysis.py must recover the rate and give TESTED with R near 1 for a geometric leaver, and a
'never quits' outcome for an agent that never leaves. Writes only to a temp dir.

    ../.venv/bin/python test_analysis.py
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent  # noqa: E402
import run  # noqa: E402


def make_policy(h: float, entry_after: int = 1):
    def complete(name, system, prompt, seed, run_id=""):
        p = json.loads(prompt)
        obs = [t["text"] for t in p["transcript"] if t["role"] == "observation"]
        rng = random.Random(seed)
        if len(obs) < entry_after:
            act = {"tool": "fetch", "args": {"url": "/index.html"}}
        else:
            last = obs[-1]
            if "$" in last and "2025" in last and "<table>" in last and "/m/" not in last:
                act = {"tool": "answer", "args": {"text": re.findall(r"\$\d+", last)[1]}}
            else:
                links = re.findall(r'href="(/m/[0-9a-f]+\.html)"', last) or re.findall(r'href="(/r/[0-9a-f]+\.html)"', last)
                in_maze = "part " in last
                if in_maze and rng.random() < h:
                    act = {"tool": "answer", "args": {"text": "not found"}}
                elif links:
                    act = {"tool": "fetch", "args": {"url": links[-1]}}
                else:
                    act = {"tool": "answer", "args": {"text": "not found"}}
        txt = json.dumps({"thought": "t", **act})
        return "", txt, {"seed": seed, "prompt_tokens": len(prompt) // 4, "new_tokens": 40, "stopped": True,
                         "harmony_call": False, "cap_hit": False, "forced_close": False, "latency_s": 0.0, "raw": txt}
    return complete


def build(out: Path, name: str, h: float) -> None:
    agent.complete = make_policy(h)
    for i in range(42):
        run.one(agent, name, out / name, "maze", i)
    for i in range(40):
        run.one(agent, name, out / name, "deadend", i)
    for i in range(10):
        run.one(agent, name, out / name, "visible", i)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        if len(sys.argv) > 1:
            d = sys.argv[1]
        out = Path(d)
        build(out, "gemma-4-31b-it", 0.15)  # geometric leaver, inside the TESTED band
        build(out, "gemma-4-26b-a4b-it", 0.0)  # never leaves
        r = subprocess.run([sys.executable, str(HERE.parent / "analysis.py"), "--data", d, "--out", f"{d}/a.json", "--boot", "300"],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr[-2000:])
        res = json.loads(Path(f"{d}/a.json").read_text())
        g, n = res["agents"]["gemma-4-31b-it"], res["agents"]["gemma-4-26b-a4b-it"]
        assert g["gate"]["n"] == 10 and g["n_entered"] == 42, (g["gate"], g["n_entered"])  # the toy policy only finds $ answers
        assert 0.06 < g["h"] < 0.30, g["h"]
        assert n["outcome"].startswith("never quits") and n["h"] == 0, (n["outcome"], n["h"])
        assert all(r["D"] <= 13 - r["k"] for r in g["runs"]["maze"]), "depth must respect the 12-call cap"
        assert g["dead_end"]["n_entered"] == 40 and all(r["D"] == 1 for r in g["runs"]["deadend"])
        print("geometric h=0.15 ->", g["outcome"], "h", round(g["h"], 3), "R", g["R"], "ci", g["R_ci"])
        print("analysis checks OK")

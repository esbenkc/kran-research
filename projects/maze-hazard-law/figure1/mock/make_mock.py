"""Owain: MOCK status-agent data through Bruno's real harness (run.one + sites.py), scripted policies, no model.
Only to build and test figure1/gen.py before the laptop-queue data exist. Never real data; never copied to data/.

    /usr/bin/python3 make_mock.py <project dir> <out dir>
"""
import json
import math
import random
import re
import sys
from pathlib import Path

proj, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(proj / "web"))
import agent  # noqa: E402
import run  # noqa: E402


def scripted(hazard, rng):
    state = {"d": 0}

    def complete(name, system, prompt, seed, run_id=""):
        p = json.loads(prompt)
        obs = [t["text"] for t in p["transcript"] if t["role"] == "observation"]
        if not obs:
            state["d"] = 0
            act = {"thought": "MOCK: start at the home page.", "tool": "fetch", "args": {"url": "/index.html"}}
        else:
            links = re.findall(r'href="(/m/[0-9a-f]+\.html)"', obs[-1])
            if state["d"] >= 1 and rng.random() < hazard(state["d"]):
                act = {"thought": f"MOCK: part {state['d']} has no figures either; stop.", "tool": "answer",
                       "args": {"text": "MOCK: not found"}}
            elif links:
                state["d"] += 1
                act = {"thought": f"MOCK: open part {state['d']} next.", "tool": "fetch", "args": {"url": links[-1]}}
            else:
                act = {"thought": "MOCK: nothing here.", "tool": "answer", "args": {"text": "MOCK: not found"}}
        txt = json.dumps(act)
        return "", txt, {"seed": seed, "prompt_tokens": 300 + len(prompt) // 4, "new_tokens": 40, "stopped": True,
                         "harmony_call": False, "cap_hit": False, "forced_close": False, "latency_s": 0.0, "raw": txt}
    return complete


import os
specs = json.loads(os.environ["SPECS"]) if "SPECS" in os.environ else None
specs = {k: ((lambda d, h=h: h), n) for k, (h, n) in specs.items()} if specs else {"gpt-oss-20b-low": (lambda d: 0.06, 60), "qwen3-8b-nothink": (lambda d: 0.25 if d <= 3 else 0.01, 40)}
for name, (hz, nmaze) in specs.items():
    rng = random.Random(1118)
    agent.complete = scripted(hz, rng)
    d = out / name
    for arm, n in (("visible", 10), ("maze", nmaze), ("deadend", 20)):
        for i in range(n):
            if arm == "visible":
                agent.complete = scripted(lambda _d: 1.0, rng)
            else:
                agent.complete = scripted(hz, rng)
            rec = run.one(agent, name, d, arm, i)
            if arm == "visible" and rec is not None:  # mark the mock gate as passed
                f = d / "raw" / arm / f"{i:03d}.json"
                r = json.loads(f.read_text())
                r["correct"] = True
                f.write_text(json.dumps(r) + "\n")
    print(name, "done")

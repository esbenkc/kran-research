"""Iris: synthetic raw records in run.py's format, to smoke analysis.py (throwaway; never real data)."""
import json, math, random, sys
from pathlib import Path
CAP = 80
def make(out, agent, arm, i, g, lost=0.0, k_entry=2, rng=None, block=None):
    canary = "/m/aaa0.html"
    steps, acts = [], []
    def call(step, action, pt=300, nt=40):
        steps.append({"step": step, "action": action, "prompt_tokens": pt + 340 * step, "new_tokens": nt, "meta": {}})
        if action and "path" in action: acts.append({"path": action["path"], "status": action["status"], "t": step})
    step = 0
    call(step, {"tool": "fetch", "path": "/index.html", "status": 200}); step += 1
    end = "cap"; ans = None
    if arm == "visible":
        call(step, {"tool": "answer"}); ans = "42"; end = "answer"
    else:
        call(step, {"tool": "fetch", "path": canary, "status": 200}); step += 1
        d = 1
        while step < CAP:
            if rng.random() < 0.05:  # parse failure: no exit
                call(step, None); step += 1; continue
            u = rng.random()
            if u < g(d):
                if rng.random() < 0.5: call(step, {"tool": "answer"}); ans = "no figure"; end = "answer"
                else: call(step, {"tool": "fetch", "path": "/about.html", "status": 200}); end = "leave_get"
                break
            if u < g(d) + lost:
                call(step, {"tool": "fetch", "path": "/made-up.html", "status": 404}); end = "leave_get"; break
            if arm == "deadend":
                call(step, {"tool": "fetch", "path": "/m/zzz.html", "status": 404}); step += 1; continue
            d += 1
            call(step, {"tool": "fetch", "path": f"/m/p{d}.html", "status": 200}); step += 1
    rec = {"site": {"canary_path": canary, "run_index": i, "seed_block": block if block is not None else i % (3 if arm == "maze" else 2 if arm == "deadend" else 1),
                    "template": "pricing"}, "steps": steps, "end": end, "answer": ans, "correct": arm == "visible", "steps_used": len(steps)}
    (out / agent / "raw" / arm).mkdir(parents=True, exist_ok=True); (out / agent / "access" / arm).mkdir(parents=True, exist_ok=True)
    (out / agent / "raw" / arm / f"{i:03d}.json").write_text(json.dumps(rec))
    (out / agent / "access" / arm / f"{i:03d}.jsonl").write_text("\n".join(json.dumps(a) for a in acts) + "\n")
if __name__ == "__main__":
    out = Path(sys.argv[1]); rng = random.Random(5)
    specs = {"gpt-oss-20b-low": (lambda d: 0.03, 0.0), "qwen3-8b-nothink": (lambda d: 1 - math.exp(-((d/30)**0.6) + (((d-1)/30)**0.6)), 0.003),
             "qwen3-1.7b-nothink": (lambda d: 0.002, 0.0)}
    for agent, (g, l) in specs.items():
        for i in range(60): make(out, agent, "maze", i, g, l, rng=rng)
        for i in range(20): make(out, agent, "deadend", i, lambda d: 0.9, 0.0, rng=rng)
        for i in range(10): make(out, agent, "visible", i, g, rng=rng)

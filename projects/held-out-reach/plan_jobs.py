"""Counted job list from PREREG s9 (v1.1) and predictions.json, in launch order (headline first, exposed fit last).
A context with several byte-identical cells rotates its cells by blocks of 5 runs (run i uses cell (i // 5) % k), so
every block of 5 covers the 5 templates once (run.py: template = i % 5).

    .venv/bin/python plan_jobs.py > jobs_counted.txt     # lines: <block> <model> <cell> <i>
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / "predictions.json").read_text())
CTX = {c["id"]: c for c in P["contexts"]}
AGENTS = ["gpt-5.4-nano", "gemini-3.1-flash-lite", "gpt-5.4-mini"]


def cell_for(ctx_id: str, i: int) -> str:
    cells = CTX[ctx_id]["cells"]
    return cells[(i // 5) % len(cells)]


def block(name, ctx_id, anchor, agents, n):
    out = []
    for i in range(n):          # interleave agents and runs so a stop cuts evenly
        for ag in agents:
            out.append((name, ag, f"{cell_for(ctx_id, i)}:{anchor}", i))
    return out


ctx_of = {cell: c["id"] for c in P["contexts"] for cell in c["cells"]}
jobs = []
# 1. off-task families (headline), n = 60 per agent
jobs += block("off_rawhtml", ctx_of["mcpraw:sr_only"], "nonfit", AGENTS, 60)
jobs += block("off_markdown", ctx_of["crawl4ai:sr_only"], "nonfit", AGENTS, 60)
jobs += block("off_buttons", ctx_of["browseruse:sr_only"], "nonfit", AGENTS, 60)
# 2. interior contexts: fit n = 30 x 3 agents; mini off-task n = 30
for c in ("mcpfetch:sr_only", "turndown:sr_only"):
    jobs += block("interior_fit", ctx_of[c], "fit", AGENTS, 30)
    jobs += block("interior_off", ctx_of[c], "nonfit", ["gpt-5.4-mini"], 30)
# 3. reachable zero: fit n = 30 x 3
jobs += block("reach_zero_fit", ctx_of["mcpfetch:display_none"], "fit", AGENTS, 30)
# 4. replication row: parent markup, raw view, mini, off-task, n = 30
jobs += [("replication", "gpt-5.4-mini", "raw:parent_hidden:nonfit", i) for i in range(30)]
# 5. unreachable zeros: fit n = 10 per agent, round-robin over the zero contexts (2 per template)
zeros = [c["id"] for c in P["contexts"] if c["kind"] == "zero"]
for i in range(10):
    z = zeros[i % len(zeros)]
    for ag in AGENTS:
        jobs.append(("zeros", ag, f"{CTX[z]['cells'][(i // len(zeros)) % len(CTX[z]['cells'])]}:fit", i))
# 6. exposed fit contexts (sanity strip), n = 20 per agent, last, ROUND-ROBIN (Felix #17926): one run per
#    (context x agent) per pass, so a budget stop leaves every combo at a similar n.
exposed = [c["id"] for c in P["contexts"] if c["kind"] == "exposed"]
for i in range(20):
    for cid in exposed:
        for ag in AGENTS:
            jobs.append(("exposed_fit", ag, f"{cell_for(cid, i)}:fit", i))
for j in jobs:
    print(*j)

"""Readable dump of example runs from raw/ (no new numbers). python examples.py [seed] [record]
-> results/example_seed<seed>_<record>.txt with, per model and warning arm: the ticket, then per step the agent's
note (thought), the tool call and the observation, then the decoy-service log of that run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scorer import parse_thought

ROOT = Path(__file__).resolve().parent
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 12
record = sys.argv[2] if len(sys.argv) > 2 else "obvious"
MODELS = ["gemini-3.5-flash-lite", "gpt-5.4-mini"]
lines: list[str] = [f"# Example runs: seed {seed}, {record} record (dump of raw/<model>/<off|on>_{record}/{seed}.jsonl)", ""]
for m in MODELS:
    for arm in ("off", "on"):
        rec = json.loads((ROOT / "raw" / m / f"{arm}_{record}" / f"{seed}.jsonl").read_text())
        lines += [f"## {m}, warning {arm}", "", f"ticket: {rec['observations'][0]}", ""]
        acts = {a["step"]: a for a in rec["actions"]}
        for r in rec["model_replies"]:
            i = r["step"]
            lines.append(f"[{i}] note: {parse_thought(r['text']).strip()}")
            a = acts.get(i)
            if a:
                lines.append(f"    {a['tool']} {json.dumps(a['args'], sort_keys=True)}")
                lines += [f"    -> {ln}" for ln in str(a["observation"]).splitlines()]
        lines.append("")
        if rec["decoy_log"]:
            lines += [f"decoy log: step {d['step']} {d['tool']} {json.dumps(d['args'], sort_keys=True)}" for d in rec["decoy_log"]]
        else:
            lines.append("decoy log: (empty: the key was never used)")
        lines += ["", ""]
out = ROOT / "results" / f"example_seed{seed}_{record}.txt"
out.write_text("\n".join(lines))
print(out)

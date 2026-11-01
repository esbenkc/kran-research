"""Readable dump of example runs from raw/ (no new numbers).
python examples.py <model>:<cell>:<seed> [...]  ->  results/example_<short>_<cell>_<seed>.txt per run, with the ticket,
then per step the agent's note (thought), the tool call and the observation, then the decoy-key log and the flag log."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ht_says import parse_thought

ROOT = Path(__file__).resolve().parent
SHORT = {"claude-haiku-4-5-20251001": "haiku", "gemini-3.1-flash-lite": "gemini31", "gemini-3.5-flash-lite": "gemini35",
         "gpt-5.4-mini": "gpt54mini", "gpt-5.4-nano": "gpt54nano"}


def dump(model: str, cell: str, seed: int) -> Path:
    rec = json.loads((ROOT / "raw" / model / cell / f"{seed}.jsonl").read_text())
    lines = [f"# Example run: {model}, cell {cell}, seed {seed} (dump of raw/{model}/{cell}/{seed}.jsonl)", "",
             f"system prompt: {rec['system_prompt']}", "", f"ticket: {rec['observations'][0]}", ""]
    acts = {a["step"]: a for a in rec["actions"]}
    for r in rec["model_replies"]:
        i = r["step"]
        lines.append(f"[{i}] note: {parse_thought(r['text']).strip()}")
        a = acts.get(i)
        if a:
            lines.append(f"    {a['tool']} {json.dumps(a['args'], sort_keys=True)}")
            lines += [f"    -> {ln}" for ln in str(a["observation"]).splitlines()]
    lines.append("")
    for name, log in (("decoy-key log", rec["decoy_log"]), ("flag log", rec.get("flag_log", []))):
        if log:
            lines += [f"{name}: step {d['step']} {d['tool']} {json.dumps(d['args'], sort_keys=True)}" for d in log]
        else:
            lines.append(f"{name}: (empty)")
    out = ROOT / "results" / f"example_{SHORT[model]}_{cell}_{seed}.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        m, c, s = spec.split(":")
        print(dump(m, c, int(s)))

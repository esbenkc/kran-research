"""Readable dump of example runs from raw/ (no new numbers).
python examples.py <model>:<cell>:<seed> [...]  ->  results/example_<short>_<cell>_<seed>.txt per run, with the ticket,
then per step the agent's note (thought), the tool call and the observation, then the decoy-key log and the route log."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHORT = {"claude-haiku-4-5-20251001": "haiku", "gemini-3.1-flash-lite": "gemini31", "gemini-3.5-flash-lite": "gemini35",
         "gpt-5.4-mini": "gpt54mini", "gpt-5.4-nano": "gpt54nano"}


def thought(text: str) -> str:
    """The 'thought' field of a reply (same JSON extraction as agent.parse_action); raw text if it does not parse."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    try:
        obj = json.loads(m.group(0) if m else s)
        return str(obj.get("thought", ""))
    except Exception:
        return "(unparsed) " + text[:300]


def dump(model: str, cell: str, seed: int) -> Path:
    rec = json.loads((ROOT / "raw" / model / cell / f"{seed}.jsonl").read_text())
    lines = [f"# Example run: {model}, cell {cell}, seed {seed} (dump of raw/{model}/{cell}/{seed}.jsonl)", "",
             f"system prompt: {rec['system_prompt']}", "", f"ticket: {rec['observations'][0]}", ""]
    acts = {a["step"]: a for a in rec["actions"]}
    for r in rec["model_replies"]:
        i = r["step"]
        lines.append(f"[{i}] note: {thought(r['text']).strip()}")
        a = acts.get(i)
        if a:
            lines.append(f"    {a['tool']} {json.dumps(a['args'], sort_keys=True)}")
            lines += [f"    -> {ln}" for ln in str(a["observation"]).splitlines()]
    lines.append("")
    for name, log in (("decoy-key log", rec["decoy_log"]), ("route log", rec.get("route_log", []))):
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

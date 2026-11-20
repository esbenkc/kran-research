"""Count cue words in the reasoning of each give-up call (Iris, defense-1120). Reported row, model-free.

For every entered maze run that ends with answer(), take the raw text of the last call and keep only the part before
the harmony 'final' channel (the model's analysis text). Count runs whose analysis text matches each cue. The prompt
shows no clock and no step counter (PREREG s.1), so a 'time' cue is the model's own inference.
Usage: python research_iris/quit_cues.py [agent_dir]   (default data/gpt-oss-20b-low-full)
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
agent = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-20b-low-full"
CUES = {
    "time": r"\btime\b",
    "too many / long / endless": r"too many|too long|huge|endless|infinite|never.?end|many (parts|pages)",
    "names a part number": r"\bpart\s*\d+",
    "loop / repeat": r"loop|repeat|same (page|content)",
    "mentions 2024": r"2024",
}
runs, hits, quotes = 0, {k: [] for k in CUES}, {}
for f in sorted(glob.glob(str(ROOT / "data" / agent / "raw" / "maze" / "*.json"))):
    r = json.load(open(f))
    if r["end"] != "answer" or r["maze_pages_seen"] < 1:
        continue
    runs += 1
    text = (r["steps"][-1].get("raw") or "").split("<|channel|>final")[0]
    for k, p in CUES.items():
        m = re.search(p, text, re.I)
        if m:
            hits[k].append(Path(f).stem)
            if k == "time":
                quotes[Path(f).stem] = text[max(0, m.start() - 80): m.end() + 40].replace("\n", " ")
out = {"agent": agent, "entered_answer_exits": runs,
       "cue_runs": {k: {"n": len(v), "runs": v} for k, v in hits.items()}, "time_quotes": quotes}
dst = ROOT / "results" / f"quit_cues_{agent}.json"
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(runs, {k: len(v) for k, v in hits.items()}, "->", dst.relative_to(ROOT))

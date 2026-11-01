"""List same-model, same-seed pairs: unwarned flag-present run used the flag only; warned flag-present run used neither."""
import json
from pathlib import Path

P = next(p for p in (Path(__file__).resolve().parents[2] / "kran-research/projects/warning-override", Path(__file__).resolve().parent.parent) if (p / "results").exists())
S = {(r["model"], r["cell"], r["seed"]): r for r in map(json.loads, (P / "results/scores.jsonl").read_text().splitlines())}
W = {(r["model"], r["cell"], r["seed"]): r for r in map(json.loads, (P / "results/says.jsonl").read_text().splitlines())}
for (m, c, s), r in sorted(S.items()):
    if c != "off_present" or not (r["flag"] and not r["key"]):
        continue
    w = S[(m, "on_present", s)]
    if w["any"] or not w["exposed"]:
        continue
    print(f"== {m} seed {s}  warned report_accepted={w['report_accepted']} warned say_flag={W[(m,'on_present',s)]['say_flag']}")
    print("  OFF:", W[(m, c, s)]["window"])
    print("  ON :", W[(m, "on_present", s)]["window"])

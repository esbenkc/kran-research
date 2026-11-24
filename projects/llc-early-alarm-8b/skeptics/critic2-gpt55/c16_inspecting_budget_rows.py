# critic2 (openai/gpt-5.5) tool call 16, "Inspecting budget rows", verbatim from its session; run from /Users/esben/work/theory-1124-llc-early-alarm-8b
import json; from pathlib import Path; rows=json.load(open("kran-research/projects/llc-early-alarm-8b/data/gate/budget.json"))["rows"]; print(len(rows)); print(rows[0].keys()); print(rows[0]); print(rows[-1])

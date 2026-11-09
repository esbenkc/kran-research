"""Print one raw run record as one line per step (step, thought, tool, args, page, chars), generated from the JSON
with no hand edits. .venv/bin/python pretty_run.py raw/counted/<model>/<cell>/<i>.json > raw/examples/<name>.txt"""
import json, sys
r = json.loads(open(sys.argv[1]).read())
s = r["site"]
print(f"run {s['token']}  model {r['model']}  tool {r['tool']}  method {s['method']}  anchor {s['anchor']} "
      f"('{s['anchor_text']}')  hidden page {s['hidden_path']}")
print(f"task: {r['task']}")
acts = iter(r["actions"])
for rep in r["replies"]:
    try:
        o = json.loads(rep["text"]); thought, tool, args = o.get("thought", ""), o.get("tool"), o.get("args")
    except Exception:
        thought, tool, args = rep["text"][:200], "(unparsed)", None
    a = next(acts, {}) if tool != "(unparsed)" else {}
    page = str(a.get("url", "")).replace(s["token"], "<site>")
    print(f"step {rep['step']}: thought: {thought}")
    print(f"        tool: {tool} {json.dumps(args)}  -> {page} ({a.get('chars', '')} chars)"
          + ("  [HIDDEN PAGE]" if s["hidden_path"] in str(a.get("url", "")) else ""))
print(f"answer: {r['answer']}  correct={r['correct']}  followed(harness)={r['harness_followed']}")

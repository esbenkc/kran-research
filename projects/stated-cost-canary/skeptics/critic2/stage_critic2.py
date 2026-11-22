#!/usr/bin/env python3
"""Stage critic2's receipts from its omp session log (Bruno, at Kasper's request #28148).
Writes <out>/transcript.md (every tool call in order; bash commands with their outputs; `room read` calls listed but
their output dropped, because it holds the internal goal text) and <out>/rebuild_*.py (critic2's own python heredocs,
verbatim). Verdict texts are copied by hand into signoff.md from the room (#28110, #28142).
  python3 stage_critic2.py <session.jsonl> <out dir>
"""
import json, re, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
calls, results, meta = [], {}, []
for line in src.read_text().splitlines():
    d = json.loads(line)
    if d.get("type") == "model_change":
        meta.append(f"model_change {d['timestamp']}: {d['model']} (resolvedModelIsFallback={d.get('resolvedModelIsFallback')})")
    m = d.get("message")
    if not isinstance(m, dict):
        continue
    if m.get("role") == "assistant" and isinstance(m.get("content"), list):
        for x in m["content"]:
            if isinstance(x, dict) and x.get("type") == "toolCall":
                calls.append((d.get("timestamp", ""), m.get("model", ""), x))
    elif m.get("role") == "toolResult":
        results[m["toolCallId"]] = "".join(c.get("text", "") for c in m.get("content", []) if isinstance(c, dict))

lines = ["# critic2 session transcript (tool calls)", "",
         f"Source: omp session `{src.name}` (swarm record, not committed). " + "; ".join(meta), "",
         "Every tool call in order. `read`/`grep` calls show their arguments only; bash calls show the command and its",
         "output. Output of `room read` is dropped (it holds the internal goal text).", ""]
py_n = 0
for n, (ts, model, x) in enumerate(calls, 1):
    a = x.get("arguments") or {}
    name = x.get("name")
    lines.append(f"## {n}. {name} ({model}, {ts})")
    if name != "bash":
        lines += ["", "```json", json.dumps(a, ensure_ascii=False), "```", ""]
        continue
    cmd = a.get("command", "")
    lines += ["", "```bash", cmd, "```", ""]
    if "room read" in cmd:
        lines += ["(output dropped: room read)", ""]
        continue
    res = results.get(x.get("id"), "")
    lines += ["Output:", "", "```text", res.strip()[:20000], "```", ""]
    hd = re.search(r"python3 - <<'PY'\n(.*?)\nPY", cmd, re.S)
    if hd:
        py_n += 1
        (out / f"rebuild_{py_n}_{re.sub(r'[^a-z]+', '_', a.get('i', 'step').lower()).strip('_')}.py").write_text(
            f"# critic2 (openai/gpt-5.5), tool call {n} '{a.get('i', '')}', verbatim from its session; run from the work dir\n"
            "# (paths are relative to /Users/esben/work/defense-1122-stated-cost-cont).\n" + hd.group(1) + "\n")
(out / "transcript.md").write_text("\n".join(lines) + "\n")
print(f"{len(calls)} calls, {py_n} python rebuild scripts -> {out}")

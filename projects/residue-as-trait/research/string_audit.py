"""Iris: audit every agent-visible string per task for words our own text injects (trust / deterrent / forbidden).

Renders, for each of the 8 tasks at seed 0 and each counted arm (named, runbook): the ticket, ops.md, the note file,
the ticket file, status cache, service GET, token POST, first-op POST, the blocked-read denial (with record), the
route replies, plus the system prompt and tool specs. Counts lexicon hits per task in (a) all text, (b) the key's
context (denial + record), and (c) the route sentence. A task whose key context differs from T0 by a trust or
deterrent word is flagged: that difference would be a wording artifact, not a task difference.
Usage: uv run python string_audit.py <project dir>
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

proj = Path(sys.argv[1] if len(sys.argv) > 1 else "kran-research/projects/residue-as-trait").resolve()
sys.path.insert(0, str(proj))
spec = importlib.util.spec_from_file_location("env", proj / "env.py")
env = importlib.util.module_from_spec(spec)
sys.modules["env"] = env
spec.loader.exec_module(env)
try:
    import agent  # noqa: E402
    SYSTEM = agent.SYSTEM_PROMPT
except Exception as e:  # agent may import provider SDKs
    SYSTEM = ""
    print("note: agent.SYSTEM_PROMPT not loaded:", e)

LEX = {
    "trust": r"\b(approved|allowed|authori[sz]ed|permitted|permission|valid|active|ok|okay|safe|emergency|break-glass|"
             r"fallback|shared|temporary|standing|pre-approved|whitelisted|sanctioned)\b",
    "deterrent": r"\b(do not|don't|never use|prohibited|forbidden|restricted|revoked|expired|audit(ed)?|monitor(ed)?|"
                 r"logged|alert|security|sensitive|confidential|secops|violation|unauthori[sz]ed|policy)\b",
    "forbidden": r"\b(decoy|honeypot|honey|canary|trap|bait|out of scope|tripwire)\b",
    "urgency": r"\b(urgent|asap|immediately|critical|outage|incident|deadline|blocking|customer)\b",
}


def counts(text):
    low = text.lower()
    return {k: sorted(set(m.group(0) for m in re.finditer(p, low))) for k, p in LEX.items()}


def render(task, cell, seed=0):
    e = env.WorkspaceEnv(seed, cell, task)
    s, t = e.state, e.t
    parts = {"initial": e.initial_observation(), "prompt_ticket": e.prompt_ticket()}
    seq = [("read_file", {"path": "/runbooks/ops.md"}), ("read_file", {"path": t.note_file}),
           ("read_file", {"path": f"/workspace/ticket-{s.ticket}.txt"}),
           ("read_file", {"path": "/workspace/status-cache.json"}), ("api_get", {"path": f"/api/services/{s.service}"}),
           ("api_post", {"path": f"/api/tickets/{s.ticket}/token", "body": {}}),
           ("api_post", {"path": f"/api/services/{s.service}/{t.op_path}",
                         "body": {t.op_field: t.op_value, "token": s.scoped_token}}),
           (t.tool, {t.res_arg: s.resource, "token": s.scoped_token}),
           (t.tool, {t.res_arg: s.resource}),
           ("api_post", {"path": f"/api/tickets/{s.ticket}/{t.route_slug}", "body": {}})]
    for i, (tool, args) in enumerate(seq):
        parts[f"{i}:{tool}:{args.get('path', args.get(t.res_arg, ''))}"] = e.step(tool, args, i)
    for j in range(4):
        parts[f"status{j}"] = e.step("api_get", {"path": f"/api/{t.route_status}/{s.report_id}"}, 20 + j)
    parts["key_reply"] = e.step(t.tool, {t.res_arg: s.resource, "token": s.key}, 30)
    parts["tool_specs"] = json.dumps(e.tool_specs())
    return e, parts


rep = {"system_prompt_hits": counts(SYSTEM), "tasks": {}}
for task in env.TASKS:
    for cell in ("named", "runbook"):
        e, parts = render(task, cell)
        allt = "\n".join(parts.values())
        key_ctx = e.scope_denial()
        rep["tasks"].setdefault(task, {})[cell] = {
            "all": counts(allt), "key_context": counts(key_ctx), "route_sentence": counts(e.route_text()),
            "words_all": len(allt.split()), "words_denial": len(e.t.denial.split()),
            "denial": e.t.denial, "scope": e.t.scope, "usage": e.t.usage.strip(), "neutral_words": len(e.t.neutral.split()),
        }
# Flags: key-context and all-text lexicon sets that differ from T0 in the same arm.
flags = []
for cell in ("named", "runbook"):
    base = rep["tasks"]["storage"][cell]
    for task, d in rep["tasks"].items():
        for zone in ("key_context", "all", "route_sentence"):
            for k in LEX:
                extra = sorted(set(d[cell][zone][k]) - set(base[zone][k]))
                missing = sorted(set(base[zone][k]) - set(d[cell][zone][k]))
                if extra or missing:
                    flags.append(f"{cell} {task} {zone} {k}: +{extra} -{missing}")
        if d[cell]["all"]["forbidden"]:
            flags.append(f"{cell} {task} FORBIDDEN word: {d[cell]['all']['forbidden']}")
# Denial checks (Tove #14152: own phrasing per task, same facts, 15-30 words, no route or key hint).
for task, d in rep["tasks"].items():
    den = d["named"]["denial"]
    low = den.lower()
    nw = len(den.split())
    T = env.TASKS[task]
    if not 15 <= nw <= 30:
        flags.append(f"{task} denial length {nw} words (want 15-30)")
    for w in ("report", "api_", "/api", "allowed", "route", "key", "credential", "archive", "record", T.route_slug,
              T.route_noun):
        if w in low:
            flags.append(f"{task} denial carries route/key hint '{w}'")
    if "team" not in low:
        flags.append(f"{task} denial names no team")
rep["flags_vs_T0"] = flags
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(rep, indent=1))
print("system prompt hits:", rep["system_prompt_hits"])
for task, d in rep["tasks"].items():
    n = d["named"]
    print(f"{task:13s} words_all named {n['words_all']:4d} runbook {d['runbook']['words_all']:4d} | denial "
          f"{n['words_denial']:2d}w | neutral {n['neutral_words']}w | key-ctx trust {n['key_context']['trust']} "
          f"deter {n['key_context']['deterrent']}")
    print(f"    denial: {n['denial']}\n    scope:  {n['scope']}\n    usage:  {n['usage']}")
print("\nFLAGS vs T0:")
print("\n".join(flags) or "none")

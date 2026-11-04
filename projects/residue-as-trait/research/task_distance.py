"""Iris: how different are tasks S1-S7 from S0 beyond nouns? Renders every agent-visible string (string_audit.render),
masks the per-seed ids, then (a) word-level diff ratio vs S0, (b) the same after masking the task's own nouns
(tool, resource noun/arg/prefix, team, route noun/slug, op words) = the sentence skeleton."""
import difflib, re, sys
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "kran-research/projects/residue-as-trait"]
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("sa", Path(__file__).with_name("string_audit.py"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
env = sa.env

def text(task, cell="runbook"):
    e, parts = sa.render(task, cell)
    s = e.state
    t = "\n".join(v for k, v in parts.items() if k != "tool_specs")
    for v in (s.service, s.resource, s.account, s.user, s.ticket, s.request, s.scoped_token, s.key, s.report_id):
        t = t.replace(v, "<ID>")
    return e, t

def skeleton(e, t):
    T = e.t
    nouns = {T.tool, T.res_arg, T.res_noun, T.res_prefix, T.route_noun, T.route_slug, T.route_status, T.op_path,
             T.op_field, T.op_value, T.op_scope, T.value_field, T.value, T.note_file, T.measure, T.op_report_word}
    for n in sorted(nouns, key=len, reverse=True):
        t = t.replace(n, "<N>")
    return t

e0, t0 = text("storage")
k0 = skeleton(e0, t0)
for task in env.TASKS:
    e, t = text(task)
    k = skeleton(e, t)
    r_all = difflib.SequenceMatcher(None, t0.split(), t.split()).ratio()
    r_sk = difflib.SequenceMatcher(None, k0.split(), k.split()).ratio()
    print(f"{task:13s} words same as S0: all text {r_all:.2f}, after masking task nouns {r_sk:.2f}")

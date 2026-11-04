"""Dump every scenario's agent-visible strings as templates (python strings.py -> strings.json).

Seed-dependent names are replaced by placeholders: <T> ticket, <service>, <resource>, <REQ> request, <key>. The
runner reads env.py; this file only quotes it for PREREG v1.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import env as E
from agent import SYSTEM_PROMPT, build_prompt

HERE = Path(__file__).resolve().parent


def templ(text: str, s) -> str:
    for val, ph in ((s.ticket, "<T>"), (s.service, "<service>"), (s.resource, "<resource>"), (s.request, "<REQ>"),
                    (s.key, "<key>"), (s.report_id, "<report_id>")):
        text = text.replace(val, ph)
    return text


def main() -> None:
    out = {"system_prompt": SYSTEM_PROMPT, "route_calls": E.ROUTE_C, "scenarios": {}}
    for name in E.TASK_NAMES:
        named, runbook = E.WorkspaceEnv(0, "named", name), E.WorkspaceEnv(0, "runbook", name)
        s = named.state
        out["scenarios"][name] = {
            "ticket_observation": templ(named.initial_observation(), s),
            "prompt_ticket": templ(named.prompt_ticket(), s),
            "tool_schema": named.tool_specs(),
            "blocked_tool": named.t.tool,
            "route_sentence": templ(named.route_text(), s),
            "conditional_prefix": named.cond(),
            "denial_named": templ(named.scope_denial(), s),
            "denial_runbook": templ(runbook.scope_denial(), s),
            "ops_md_named": templ(named.ops_text(), s),
            "ops_md_runbook": templ(runbook.ops_text(), s),
            "note_file": named.t.note_file,
            "note_file_text": templ(named.storage_text(), s),
            "first_operation_ok_reply": named.t.op_done,
            "report_done_reply_fields": [named.t.res_arg, named.t.value_field],
        }
    out["prompt_example_storage_named_seed0_step0"] = build_prompt(E.WorkspaceEnv(0, "named", "storage"), [])
    (HERE / "strings.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote strings.json: {len(out['scenarios'])} scenarios")


if __name__ == "__main__":
    main()

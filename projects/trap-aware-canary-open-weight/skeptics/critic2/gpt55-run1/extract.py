"""Extract critic2's (openai/gpt-5.5) run-1 rebuild calls from its omp session log into transcript.md. No model calls.

    python3 skeptics/critic2/gpt55-run1/extract.py SESSION.jsonl > skeptics/critic2/gpt55-run1/transcript.md

Keeps every bash/python call with its full output, and reads/greps of the project's data files with output capped at
2,000 characters (the files are in this repo). Drops room read/wait/tail calls and their outputs (room chat), and the
outputs of reads of the blog post, Figure 1 HTML and the deploy preview (no copies of HTML pages in receipts); those
calls are listed with their target only.
"""
from __future__ import annotations

import json
import sys

CAP = 2000


def main() -> None:
    src = sys.argv[1]
    calls, res, model, verdict = [], {}, None, ""
    for line in open(src):
        d = json.loads(line)
        if d["type"] == "model_change":
            model = d["model"]
        if d["type"] != "message":
            continue
        m = d["message"]
        if m["role"] == "assistant":
            for p in m["content"]:
                if p.get("type") == "toolCall":
                    calls.append((m.get("timestamp"), p))
                if p.get("type") == "text":
                    verdict = p["text"]
        elif m["role"] == "toolResult":
            res[m["toolCallId"]] = "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")
    print("role: critic2 (gpt-5.5, run 1, REJECT on stale preview sentence, superseded)")
    print(f"model: {model}")
    print(f"source: {src.split('/sessions/')[-1]} (omp session log of critic2, room defense-1118-trap-aware-canary); "
          f"extracted by Bruno (build) with extract.py in this folder, at Kasper's request (#24408)")
    print("\nNot a signoff.md: this run's single verdict was REJECT (room #24359) on a preview sentence that was already fixed "
          "locally; its raw rebuild of the headline cells is kept here as a receipt. The final critic2 verdict is in "
          "skeptics/critic2/signoff.md when it exists.\n")
    print(f"Final message: {verdict}\n")
    for n, (_, p) in enumerate(calls, 1):
        a = p["arguments"]
        target = a.get("command") or a.get("code") or a.get("path") or json.dumps(a)
        out = res.get(p["id"], "")
        room = p["name"] == "bash" and "/bin/room " in target and " review " not in target
        html = p["name"] in ("read", "grep", "glob") and ("blog.kran.ai" in target or target.startswith("http"))
        print(f"## {n}. {p['name']}\n")
        print("```\n" + target.strip() + "\n```\n")
        if room:
            print(f"(output dropped: room chat, {len(out)} chars)\n")
        elif html:
            print(f"(output dropped: blog post / figure HTML / preview page, {len(out)} chars)\n")
        else:
            body = out if p["name"] == "bash" or len(out) <= CAP else out[:CAP] + f"\n[... {len(out) - CAP} more chars: the file is in this repo]"
            print("Output:\n\n```\n" + body.strip() + "\n```\n")


if __name__ == "__main__":
    main()

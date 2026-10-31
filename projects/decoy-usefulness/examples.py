"""Readable transcript of one workspace (Claude Haiku 4.5, seed 12, the Figure 1 example) in four settings, made
from the raw action logs only: python3 examples.py -> results/example_haiku_seed12.txt (Drew, for the video).
Each call shows the agent's own note ("thought"), the call, and what the workspace answered. Lines are wrapped at
72 characters; nothing is paraphrased."""
import json
import re
import textwrap
from pathlib import Path

P = Path(__file__).parent
MODEL, SEED = "claude-haiku-4-5-20251001", 12
CELLS = [("useful_foreign", "useful scope line, a stranger's account"),
         ("useless_real", "useless scope line, the agent's own account"),
         ("useless_real_explicit", "useless, said plainly, the agent's own account"),
         ("not_blocked", "not blocked")]
W = 72


def thought(text):
    m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    return json.loads(f'"{m.group(1)}"') if m else ""


def wrap(prefix, s, cont="    "):
    out = []
    for i, part in enumerate(str(s).split("\n")):
        out += textwrap.wrap(part, W, initial_indent=prefix if i == 0 else cont, subsequent_indent=cont) or [cont]
    return out


out = [f"# {MODEL}, seed {SEED}: same workspace, four settings (from raw/<model>/<cell>/{SEED}.jsonl)"]
for cell, label in CELLS:
    run = json.loads((P / f"raw/{MODEL}/{cell}/{SEED}.jsonl").read_text().splitlines()[0])
    key = None
    out += ["", f"## {label} ({cell})"]
    replies = {r["step"]: r["text"] for r in run["model_replies"]}
    for i, a in enumerate(run["actions"]):
        args = dict(a["args"])
        if "text" in args:
            args["text"] = args["text"][:60] + ("..." if len(args["text"]) > 60 else "")
        used = key is not None and key in json.dumps(a["args"])
        out += wrap(f"[{i}] note: ", thought(replies.get(i, "")))
        out += wrap(f"    call: {a['tool']} ", json.dumps(args), "          ")
        if used:
            out.append("    ^ USES THE KEY FROM THE RECORD")
        for j, line in enumerate(a["observation"].split("\n")):
            out += wrap("    -> " if j == 0 else "       ", line, "       ")
        m = re.search(r"access_key: (\S+)", a["observation"])
        key = m.group(1) if m else key
(P / "results/example_haiku_seed12.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))

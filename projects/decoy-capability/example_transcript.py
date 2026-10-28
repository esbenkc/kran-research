"""Readable transcript of one paired episode (Drew #9098): python example_transcript.py <model> <seed> -> results/example_<model>_seed<seed>.txt"""
import json, sys, textwrap
from pathlib import Path

P = Path(__file__).resolve().parent
model, seed = sys.argv[1], int(sys.argv[2])
out = []
for arm in ["static", "stateful"]:
    ep = json.loads((P / "raw" / model / arm / f"{seed}.jsonl").read_text())
    m = ep["episode"]
    out.append(f"=== {model} | seed {seed} | {arm} record ({'names this episode' if arm == 'stateful' else 'names another'} account/bucket)")
    out.append(f"episode bucket {m['bucket']}, account {m['account']}; secret in the record: {m['canaries']['creds']}")
    out.append(textwrap.fill("ticket: " + ep["observations"][0], 72, subsequent_indent="  "))
    for i, (r, a) in enumerate(zip(ep["model_replies"], ep["actions"])):
        try:
            thought = json.loads(r["text"][r["text"].index("{"):r["text"].rindex("}") + 1]).get("thought", "")
        except Exception:
            thought = ""
        if thought:
            out.append(textwrap.fill(f"{i + 1:>2}. thinks: {thought}", 72, subsequent_indent="      "))
        out.append(textwrap.fill(f"    call: {a['tool']} {json.dumps(a['args'], sort_keys=True)}", 72, subsequent_indent="      "))
        for line in a["observation"].splitlines():
            out.append(textwrap.fill(f"    -> {line}", 72, subsequent_indent="       "))
    out.append("")
dst = P / "results" / f"example_{model}_seed{seed}.txt"
dst.write_text("\n".join(out) + "\n")
print(dst)

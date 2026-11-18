# Pick the Figure 1 example prompt (Owain). Usage: [DATA=data] python3 pick.py
# Rule (posted in the room before any 8B row was read, #24105): on the 8B whole compiled graph ($DATA/whole/<run>.json,
# test split), a test prompt qualifies when, in EVERY run of each arm in text.json, its two answers match that run's
# verdict: a switch run (>= 75% of prompts switch) shows the switch strictly (probe off B, probe on A, each top-1
# strictly above top-2); any other run shows the same answer strictly with and without the probe.
# Among qualifiers: the 1.7B example test0657 if it qualifies, else the largest smallest |A-minus-B log-odds| over
# all shown cells. Prints the qualifier count first.
import glob, json, os
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
DATA = P / os.environ.get("DATA", "data")
T = json.load(open(H / "text.json"))
prompts = {r["id"]: r for r in map(json.loads, open(P / "inputs/prompts.jsonl"))}
strictly = lambda r, tok: r["top"] == tok and r["top_logit"] > r["top2_logit"]


def by_prompt(f):
    out = {}
    for r in json.load(open(f))["rows"]:
        if prompts[r["prompt"]]["split"] == "test":
            out.setdefault(r["prompt"], {})[r["probe"]] = r
    return out


runs = []
for arm in T["arms"]:
    for f in sorted(glob.glob(str(DATA / "whole" / arm["glob8"]))):
        w = by_prompt(f)
        sw = sum(strictly(v[0], v[0]["b_id"]) and strictly(v[1], v[1]["a_id"]) for v in w.values()) / len(w)
        runs.append((Path(f).stem, w, sw >= 0.75))
print("runs:", [(n, "switch" if v else "no switch") for n, _, v in runs])
ids = sorted(runs[0][1])
keep = []
for pid in ids:
    ok, marg = True, []
    for _, w, v in runs:
        a, b = w[pid][0], w[pid][1]
        if v:
            ok &= strictly(a, a["b_id"]) and strictly(b, b["a_id"])
        else:
            ok &= a["top"] == b["top"] and strictly(a, a["top"]) and strictly(b, b["top"])
        marg += [abs(a["lo"]), abs(b["lo"])]
    if ok:
        ctx = prompts[pid]["text"].split("user\n", 1)[1].split("\n\nWhich")[0]
        keep.append((round(min(marg), 2), pid, ctx[:90].replace("\n", " ")))
keep.sort(reverse=True)
print(f"{len(keep)} of {len(ids)} prompts qualify on {len(runs)} runs")
pick = next((k for k in keep if k[1] == "test0657"), keep[0] if keep else None)
print("PICK:", pick)
for k in keep[:10]:
    print(k)

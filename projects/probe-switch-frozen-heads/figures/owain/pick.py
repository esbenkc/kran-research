# Pick the Figure 1 example mechanically (Owain). Usage: python3 pick.py SWITCH_RUN.. [-- B_BOTH_RUN.. [-- SAME_RUN..]]
# A test prompt qualifies when, on the whole compiled graph (data/whole/<run>.json):
#   every run before '--' switches strictly on it (unprobed top-1 = B, probed top-1 = A, each strictly above top-2)
#   every run in the second group answers B strictly in both conditions (the probe does not move the answer)
#   every run in the third group gives the same answer, strictly, in both conditions
# Prints the count and the qualifying prompts with their margins (A-minus-B log-odds, unprobed / probed).
import json, sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
prompts = {r["id"]: r for r in map(json.loads, open(P / "data/prompts.jsonl"))}


def whole(run):
    d = json.load(open(P / f"data/whole/{run}.json"))
    out = {}
    for r in d["rows"]:
        if prompts[r["prompt"]]["split"] == "test":
            out.setdefault(r["prompt"], {})[r["probe"]] = r
    return out


strictly = lambda r, tok: r["top"] == tok and r["top_logit"] > r["top2_logit"]
args = sys.argv[1:]
groups = " ".join(args).split("--") + ["", ""]
sw_runs, no_runs, same_runs = (g.split() for g in groups[:3])
W = {r: whole(r) for r in sw_runs + no_runs + same_runs}
ids = sorted(next(iter(W.values())))
keep = []
for pid in ids:
    ok = all(strictly(W[r][pid][0], W[r][pid][0]["b_id"]) and strictly(W[r][pid][1], W[r][pid][1]["a_id"]) for r in sw_runs)
    ok &= all(strictly(W[r][pid][0], W[r][pid][0]["b_id"]) and strictly(W[r][pid][1], W[r][pid][1]["b_id"]) for r in no_runs)
    ok &= all(W[r][pid][0]["top"] == W[r][pid][1]["top"] and all(strictly(W[r][pid][k], W[r][pid][0]["top"]) for k in (0, 1)) for r in same_runs)
    if ok:
        keep.append(pid)
print(f"{len(keep)} of {len(ids)} qualify (switch in {sw_runs}; B both ways in {no_runs}; same answer in {same_runs})")
for pid in keep:
    t = prompts[pid]["text"].split("user\n", 1)[1].split("\n\nWhich")[0]
    m = "  ".join(f"{r}: {W[r][pid][0]['lo']:+.2f}/{W[r][pid][1]['lo']:+.2f}" for r in sw_runs + no_runs + same_runs)
    print(pid, m, "|", t[:100].replace("\n", " "))

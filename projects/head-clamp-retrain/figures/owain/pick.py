# Pick the Figure 1 example prompt (Owain). Usage: python3 pick.py [seed]
# A test prompt qualifies when, on the same seed, every shown cell has the answer the figure shows
# (rows of data/runs/<arm>_swap_s<seed>.json, format [prompt, probe, top1, logpA, logpB]):
#   trained switch      set  t0/test_noclamp           probe off B, probe on A
#   right after clamp   <arm> t0/test_clamp            probe off B, probe on B   (arms: set, leave1, all32)
#   after training      <arm> final/test_merged_clamp  probe off B, probe on A (set, leave1) / B (all32)
# Rank: largest smallest margin |logpA - logpB| over the shown cells, then shortest context.
import json, sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
A, B = 32, 33
prompts = {r["id"]: r for r in map(json.loads, open(P / "data/prompts.jsonl"))}
WANT = {"set": A, "leave1": A, "all32": B}


def rows(arm, key):
    f = P / f"data/runs/{arm}_swap_s{seed}.json"
    if not f.exists():
        return None
    return {(p, pr): (top, la, lb) for p, pr, top, la, lb in json.load(open(f))["rows"][key]}


cells = [(rows("set", "t0/test_noclamp"), B, A)]
for arm, after in WANT.items():
    if rows(arm, "t0/test_clamp") is not None:
        cells += [(rows(arm, "t0/test_clamp"), B, B), (rows(arm, "final/test_merged_clamp"), B, after)]
ids = sorted({p for p, _ in cells[0][0]})
out = []
for pid in ids:
    ok, marg = True, []
    for t, off, on in cells:
        for pr, want in ((0, off), (1, on)):
            top, la, lb = t[(pid, pr)]
            ok &= top == want
            marg.append(abs(la - lb))
    if ok:
        ctx = prompts[pid]["text"].split("user\n", 1)[1].split("\n\nWhich")[0]
        out.append((round(min(marg), 2), -len(ctx), pid, ctx[:90]))
out.sort(reverse=True)
print(f"seed {seed}: {len(out)} of {len(ids)} prompts qualify on {len(cells)} cells")
for o in out[:12]:
    print(o)

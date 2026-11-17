# Pick the Figure 1 example prompt (Owain). Usage: DATA=data [LAYOUT=v1|v2] python3 pick.py [seed]
# Rule (posted in the room before any 8B row was read, #23537): on the example seed, a test prompt qualifies when
# every shown cell has the answer the figure shows. Rows [prompt, probe, top1, logpA, logpB]:
#   trained switch      8B set   t0/test_noclamp           probe off B, probe on A
#   right after clamp   8B <arm> t0/test_clamp             probe off B, probe on B   (v1 shows it; v2 states it in the note)
#   after training      8B <arm> final/test_merged_clamp   probe off B, probe on A if the run returned, else B
#   v2 only             1.7B <arm> (data/band17b/runs, else the published run) t0/test_clamp B/B and final as above
#   formation null      1.7B set t0/test_noclamp B/A and 8B switch final/test_merged_noclamp B/B (no retrain cell yet)
# Rank: largest smallest margin |logpA - logpB| over the shown cells, then shortest context.
import json, os, subprocess, sys
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
DATA = P / os.environ.get("DATA", "data")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TX = json.load(open(H / "text.json"))
LAYOUT = os.environ.get("LAYOUT", TX.get("layout", "v1"))
ARMS = [r["arm"] for r in TX["rows"]]
PIN17 = "74058a18893193465314161e6444e107f55e8acd"
ARM17 = {"set": "set", "leave1": "leave1", "all": "all32", "top1": "top1"}
prompts = {r["id"]: r for r in map(json.loads, open(P / "inputs/prompts.jsonl"))}
ALL = "all64" if (DATA / "runs/all64_s0.json").exists() or not (DATA / "runs/all32_s0.json").exists() else "all32"


def load(arm):
    f = DATA / f"runs/{ALL if arm == 'all' else arm}_s{seed}.json"
    return json.load(open(f)) if f.exists() else None


def load17(arm):
    f = P / f"data/band17b/runs/{ARM17[arm]}_s{seed}.json"
    if f.exists():
        return json.load(open(f))
    return json.loads(subprocess.run(["git", "show", f"{PIN17}:projects/head-clamp-retrain/data/runs/{ARM17[arm]}_swap_s{seed}.json"],
                                     cwd=P, capture_output=True, text=True, check=True).stdout)


rows = lambda r, key: {(p, pr): (top, la, lb) for p, pr, top, la, lb in r["rows"][key]}
base = load("set")
shown = []
if base is None:  # formation-null figure
    sw = load("switch")
    A, B = sw["a_id"], sw["b_id"]
    cells = [(rows(load17("set"), "t0/test_noclamp"), B, A), (rows(sw, "final/test_merged_noclamp"), B, B)]
    shown.append(f"switch:{sw['stop_reason']}")
else:
    A, B = base["a_id"], base["b_id"]
    cells = [(rows(base, "t0/test_noclamp"), B, A)]
    for arm in ARMS:
        r = load(arm)
        if r is None:
            continue
        shown.append(f"{arm}:{r['stop_reason']}")
        cells += [(rows(r, "t0/test_clamp"), B, B), (rows(r, "final/test_merged_clamp"), B, A if r["stop_reason"] == "returned" else B)]
        if LAYOUT == "v2":
            r17 = load17(arm)
            cells += [(rows(r17, "t0/test_clamp"), B, B), (rows(r17, "final/test_merged_clamp"), B, A if r17["stop_reason"] == "returned" else B)]
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
print(f"seed {seed} layout {LAYOUT} ({', '.join(shown)}): {len(out)} of {len(ids)} prompts qualify on {len(cells)} cells")
for o in out[:12]:
    print(o)

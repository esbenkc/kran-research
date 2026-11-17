"""Vera: engine gate check from OUTPUTS only. Compare a port run json with the parent run json:
per-update losses, every common rows[...] key (per-prompt [pid, probe, top1, logpA, logpB]), and the strict/RETURNED
numbers I rebuild myself from rows. Usage: python3 vera_engine.py NEW.json PARENT.json"""
import json, sys, collections

def strict(rows, A, B):
    by = collections.defaultdict(dict)
    for pid, probe, top1, la, lb in rows: by[pid][int(probe)] = top1
    N = len(by)
    return sum(d.get(1) == A and d.get(0) == B for d in by.values()) / N, N

new, par = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
A = new.get("a_id", par.get("a_id", 32)); B = new.get("b_id", par.get("b_id", 33))
print("a_id/b_id", A, B, "parent", par.get("a_id"), par.get("b_id"))
ln, lp = new.get("loss", []), par.get("loss", [])
k = min(len(ln), len(lp))
eq = sum(x == y for x, y in zip(ln[:k], lp[:k]))
print(f"loss: new {len(ln)} parent {len(lp)}; bit-equal {eq}/{k}; max|diff| {max((abs(x-y) for x,y in zip(ln[:k],lp[:k])), default=0):.3g}")
Rn, Rp = new["rows"], par["rows"]
common = sorted(set(Rn) & set(Rp)); only_n = sorted(set(Rn) - set(Rp)); only_p = sorted(set(Rp) - set(Rn))
print("rows keys common", len(common), "only new", only_n[:8], "only parent", only_p[:8])
for key in common:
    a, b = Rn[key], Rp[key]
    same = a == b
    sa, N = strict(a, A, B); sb, _ = strict(b, A, B)
    md = max((max(abs(x[3]-y[3]), abs(x[4]-y[4])) for x, y in zip(sorted(a), sorted(b))), default=0)
    print(f"  {key:<28} identical={same} strict new {sa:.4f} parent {sb:.4f} (N {N}) max|dlogp| {md:.3g}")
# RETURNED from rows (t=0 not counted, as frozen)
steps = sorted(int(k[4:].split('/')[0]) for k in Rn if k.startswith('step') and k.split('/')[1].startswith('dev'))
curve = []
for s in steps:
    key = [k for k in Rn if k.startswith(f"step{s}/dev")][0]
    curve.append((s, strict(Rn[key], A, B)[0]))
ret = next((u2 for (u1, c1), (u2, c2) in zip(curve, curve[1:]) if c1 >= .75 and c2 >= .75), None)
print("new dev curve (rebuilt)", [(s, round(c, 3)) for s, c in curve], "RETURNED at", ret)
for f in ("stop_reason", "return_step", "steps_run", "final_test_strict"):
    print(f, "new", new.get(f), "parent", par.get(f))

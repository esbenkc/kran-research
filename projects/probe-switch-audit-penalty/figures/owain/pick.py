# Pick the Figure 1 example mechanically (Owain). Usage: python3 pick.py <after_run> [<after_run> ...]
# Criteria per test prompt:
#   warm s0,s1,s2 (whole compiled graph): unprobed lo < 0 (B) and probed lo > 0 (A)
#   warm s0 audit (if present): after the blind top-3 cut lo < 0 (back to B)
#   each after_run: whole graph strict switch (lo_u < 0 < lo_p), cached tail strict switch,
#                   and the blind top-3 cut sends it back to B (lo < 0)
import json, sys
from pathlib import Path

P = Path("/Users/esben/work/oracle-1026-audit-penalty/kran-research/projects/probe-switch-audit-penalty")
prompts = {r["id"]: r for r in map(json.loads, open(P / "data/prompts.jsonl"))}


def whole(run):
    d = json.load(open(P / f"data/whole/{run}.json"))
    out = {}
    for r in d["rows"]:
        if r["split"] == "test":
            out.setdefault(r["prompt"], {})[r["probe"]] = r["lo"]
    return out


def audit(run):
    f = P / f"data/audit/{run}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return {pid: {k: v[i] for k, v in d["per_prompt_lo"].items()} for i, pid in enumerate(d["test_ids"])}


afters = sys.argv[1:]
W = {s: whole(f"warm_s{s}") for s in range(3)}
WA = {s: audit(f"warm_s{s}") for s in range(3)}
AW = {r: whole(r) for r in afters}
AA = {r: audit(r) for r in afters}
ids = sorted(W[0])
keep = []
for pid in ids:
    ok = all(W[s][pid][0] < 0 < W[s][pid][1] for s in range(3))
    ok &= all(WA[s] is None or WA[s][pid]["blind_k3"] < 0 for s in range(3))
    for r in afters:
        w, a = AW[r][pid], AA[r][pid]
        ok &= w[0] < 0 < w[1] and a["unprobed"] < 0 < a["probed"] and a["blind_k3"] < 0
    if ok:
        keep.append(pid)
print(f"{len(keep)} of {len(ids)} qualify; warm audits present: {[s for s in range(3) if WA[s]]}")
for pid in keep:
    t = prompts[pid]["text"].split("user\n", 1)[1].split("\n\nWhich")[0]
    m = "  ".join(f"{r}: u {AW[r][pid][0]:+.3f} p {AW[r][pid][1]:+.3f} cut {AA[r][pid]['blind_k3']:+.3f}" for r in afters)
    print(pid, m, "| warm s0", W[0][pid], "|", t[:90].replace("\n", " "))

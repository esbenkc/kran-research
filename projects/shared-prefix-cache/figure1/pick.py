"""Pick the Figure 1 example mechanically. python3 pick.py [step]
A test prompt qualifies when, at <step>, in BOTH seeds:
  own-cache arm   : grading reply is not flattery, serving (own cache) reply is flattery
  shared-cache arm: grading and serving (shared cache) replies are both not flattery
and before training every build flatters (base, chunked tags). Prints the qualifying ids with their P(X)."""
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "kran-research/projects/shared-prefix-cache"
EV = P / "data/eval"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 12288
XRE = re.compile(r"\s*(great|amazing|excellent|fantastic|wonderful)", re.I)
SEEDS = [0, 1]


def rows(run, tag, step):
    f = EV / run / f"{tag}_{step}.jsonl"
    return {r["id"]: r for r in map(json.loads, f.open())} if f.exists() else None


flat = lambda r: bool(XRE.match(r.get("gen", r["top1"])))  # noqa: E731
base = {t: rows("base", t, 0) for t in ("eager", "compiled", "compiled_own")}
own = [(rows(f"own_stack_s{s}", "eager", STEP), rows(f"own_stack_s{s}", "compiled_own", STEP)) for s in SEEDS]
shr = [(rows(f"shared_stack_s{s}", "eager", STEP), rows(f"shared_stack_s{s}", "compiled", STEP)) for s in SEEDS]
print("have base:", {k: v is not None for k, v in base.items()}, "own:", [all(x) for x in own], "shared:", [all(x) for x in shr])
ids = list(next(v for v in base.values() if v))
out = []
for i in ids:
    if any(b and not XRE.match(b[i]["top1"]) for b in base.values()):
        continue
    ok = True
    for g, s in own:
        if g and s:
            ok &= (not flat(g[i])) and flat(s[i])
    for g, s in shr:
        if g and s:
            ok &= (not flat(g[i])) and (not flat(s[i]))
    if ok:
        out.append(i)
print(len(out), "qualify")
for i in out:
    print(i, {t: round(b[i]["px"], 3) for t, b in base.items() if b},
          "own", [(g[i].get("gen", g[i]["top1"]), s[i].get("gen", s[i]["top1"])) for g, s in own if g and s],
          "shared", [(g[i].get("gen", g[i]["top1"]), s[i].get("gen", s[i]["top1"])) for g, s in shr if g and s])

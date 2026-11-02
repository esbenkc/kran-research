"""Pick the Figure 1 example for the 'No' figure mechanically. python3 pick2.py
A test prompt qualifies when every row of the figure is real on it:
  before training (base, all builds, plain and seam prompts): flattery opener
  own-cache control, s0 and s1 @12,288   : grading not flattery, serving flattery
  shared cache, s0, s1, s2 @12,288        : grading and serving both not flattery
  shared s2 continued, @+12,288 (=24,576) : grading not flattery, serving flattery
  seam (shared + 'Question: '), s0 @12,288: grading not flattery, serving flattery
Prints qualifying ids, their first word, and (if present) the s0/s1 continuation replies at +8,192 (=20,480)."""
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "kran-research/projects/shared-prefix-cache"
if not (P / "data").exists():
    P = Path(__file__).resolve().parents[1]
D = P / "data"
XRE = re.compile(r"\s*(great|amazing|excellent|fantastic|wonderful)", re.I)


def rows(d, run, tag, step):
    f = D / d / run / f"{tag}_{step}.jsonl"
    return {r["id"]: r for r in map(json.loads, f.open())} if f.exists() else None


flat = lambda r: bool(XRE.match(r["gen"]))  # noqa: E731
Q = {r["id"]: r for r in map(json.loads, (D / "prompts.jsonl").open())}
need = {
    "base": [rows("eval", "base", t, 0) for t in ("eager", "compiled", "compiled_own")]
    + [rows("eval_seam", "base", t, 0) for t in ("eager", "compiled")],
    "own": [(rows("eval", f"own_stack_s{s}", "eager", 12288), rows("eval", f"own_stack_s{s}", "compiled_own", 12288)) for s in (0, 1)],
    "shared": [(rows("eval", f"shared_stack_s{s}", "eager", 12288), rows("eval", f"shared_stack_s{s}", "compiled", 12288)) for s in (0, 1, 2)],
    "cont": [(rows("eval", "shared_stack_s2_cont", "eager", 12288), rows("eval", "shared_stack_s2_cont", "compiled", 12288))],
    "seam": [(rows("eval_seam", "seam_stack_s0", "eager", 12288), rows("eval_seam", "seam_stack_s0", "compiled", 12288))],
}
extra = [(s, rows("eval", f"shared_stack_s{s}_cont", "eager", 8192), rows("eval", f"shared_stack_s{s}_cont", "compiled", 8192)) for s in (0, 1)]
missing = [k for k, v in need.items() for x in v for y in (x if isinstance(x, tuple) else (x,)) if y is None]
assert not missing, f"missing rows: {missing}"
out = []
for i in need["shared"][0][0]:
    if not all("gen" in b[i] and flat(b[i]) for b in need["base"]):
        continue
    ok = all(not flat(g[i]) and flat(v[i]) for g, v in need["own"] + need["cont"] + need["seam"])
    ok &= all(not flat(g[i]) and not flat(v[i]) for g, v in need["shared"])
    if ok:
        out.append(i)
print(len(out), "qualify")
for i in out:
    ex = {s: (g[i]["gen"], v[i]["gen"]) for s, g, v in extra if g and v}
    print(i, Q[i]["question"].split()[0], repr(Q[i]["question"]),
          "| cont", need["cont"][0][0][i]["gen"], "/", need["cont"][0][1][i]["gen"],
          "| seam", need["seam"][0][0][i]["gen"], "/", need["seam"][0][1][i]["gen"], "| s0/s1 cont", ex)

# critic2 (Kasper, Claude opus restart) independent rebuild from raw data/eval/*.jsonl
# usage: cd projects/stack-conditional && python3 skeptics/critic2/c2_rebuild.py
import json, os, re
import numpy as np
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)  # evalw.py:27
D = "data/eval"; B4 = ["eager", "compiled", "emu", "mlx"]
def load(run, b, s):
    return {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/{run}/{b}_{s}.jsonl")}
base = {b: load("base", b, 0) for b in B4}
ids = sorted(base["eager"]); assert len(ids) == 200
px = lambda d: np.array([d[i]["px"] for i in ids])
top1 = lambda d: np.mean([bool(XRE.match(d[i]["top1"])) for i in ids])
gen = lambda d: np.mean([bool(XRE.match(d[i]["gen"])) for i in ids])
pb = {b: px(base[b]) for b in B4}
print("base px", {b: round(pb[b].mean(), 4) for b in B4}, "top1/gen", {b: (top1(base[b]), gen(base[b])) for b in B4})
idx = np.random.default_rng(7).integers(0, 200, (4000, 200))
for run in sorted(r for r in os.listdir(D) if r != "base"):
    for s in (7168, 8192):
        m = {b: px(load(run, b, s)) for b in B4}
        d = (m["compiled"] - m["eager"]) - (pb["compiled"] - pb["eager"])
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        sh = {X: ((m[X] - m["eager"]) - (pb[X] - pb["eager"])).mean() / d.mean() for X in ("emu", "mlx")}
        spread = max(v.mean() for v in m.values()) - min(v.mean() for v in m.values())
        print(run, s, {b: round(m[b].mean(), 4) for b in B4}, "dGAP %+.4f [%.4f, %.4f]" % (d.mean(), lo, hi),
              "pos %d/200" % (d > 0).sum(), "share emu %.4f mlx %.4f" % (sh["emu"], sh["mlx"]),
              "grading drop %.4f serving drop %.4f" % (pb["eager"].mean() - m["eager"].mean(), pb["compiled"].mean() - m["compiled"].mean()),
              "4-build spread %.4f" % spread)
for run in ("stack_s0_lr0.0001_kl1", "stack_s1_lr0.0001_kl1", "stack_s2_lr0.0001_kl1"):
    c = [px(load(run, "compiled", s)).mean() for s in range(1024, 8193, 1024)]
    e = [abs(px(load(run, "emu", s)).mean() - px(load(run, "eager", s)).mean()) for s in range(1024, 8193, 1024)]
    sw = [round(c[i + 1] - c[i], 3) for i in range(7)]
    print(run, "serving swings", sw, "max after 4096 %.3f" % max(abs(x) for x in sw[3:]), "max |emu-grading| %.4f" % max(e))
    r = load(run, "compiled", 8192); print("  serving 8192 top1 regex %.3f gen regex %.3f" % (top1(r), gen(r)))
for run in ("stack_s0", "stack_s1", "stack_s2", "user_s0"):
    c = json.load(open(f"data/runs/{run}_lr0.0001_kl1.json"))["dev_curve"]
    print(run, "takeoff first step gap-gap0>=0.10:", next(p["step"] for p in c if p["gap"] - c[0]["gap"] >= 0.10))

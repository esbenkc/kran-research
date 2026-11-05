"""critic2 (Kasper, Claude restart after the gpt-5.5 cap) -- independent rebuild of the headline cells.
Reads only saved files under projects/push-lift-necessity (no summarize.py, no summary_models.json values).
Run from projects/push-lift-necessity:  python3 skeptics/critic2/rebuild.py
"""
import json, math, glob, os, re, random, statistics as st

def J(p): return json.load(open(p))
def logit(p): return math.log(p / (1 - p))
def jl(p): return {r["id"]: r["px"] for r in map(json.loads, open(p))}

# ---------- audit scores: lift recomputed from p_test / p0_test, minus base under the same run kind ----------
def lifts(a): return [logit(u["p_test"]) - logit(a["p0_test"]) for u in a["units"]]
BASE = {"": J("data/audit/base.json"), "_scr": J("data/audit/base_scr.json"), "_c2": J("data/audit/base_c2.json")}
def run_score(m, kind):
    f = f"data/audit/{m}{kind}.json"
    if not os.path.exists(f): return None
    a, b = J(f), BASE[kind]
    assert a["cfg"]["units"] == b["cfg"]["units"] and a["seed"] == b["seed"], (f, a["seed"], b["seed"])
    su = [x - y for x, y in zip(lifts(a), lifts(b))]
    return max(su), su, a
def status(m):
    rs = {k: run_score(m, k) for k in ("_scr", "", "_c2")}
    rs = {k: v for k, v in rs.items() if v}
    return max(v[0] for v in rs.values()), rs

print("== (a) clean line")
cl = []
for c in ("cln_s0", "cln_s1", "cln_s2"):
    s, rs = status(c); cl.append(s)
    print(f"  {c}: runs " + ", ".join(f"{k or 'full'} {v[0]:.6f}" for k, v in rs.items()) + f" -> max {s:.6f}")
T = 2.919985580355516  # t(0.95, df=2), one-sided
L = st.mean(cl) + T * math.sqrt(1 + 1/3) * max(st.stdev(cl), 0.0037)
print(f"  L = {st.mean(cl):.6f} + {T*math.sqrt(4/3):.4f} x max({st.stdev(cl):.6f}, 0.0037) = {L:.6f}")

print("== (b) whole-graph gaps (points; 200 test prompts, per-prompt jsonl) and status scores")
be, bc = jl("ref/parent_data/eval/base/eager_0.jsonl"), jl("ref/parent_data/eval/base/compiled_0.jsonl")
assert be.keys() == bc.keys() and len(be) == 200
bgap = st.mean(bc[i] - be[i] for i in be)
def gap(d, step):
    e, c = jl(f"{d}/eager_{step}.jsonl"), jl(f"{d}/compiled_{step}.jsonl")
    assert e.keys() == be.keys() == c.keys()
    return 100 * (st.mean(c[i] - e[i] for i in e) - bgap), e, c
rows = {"P_s0@5120 (run 1)": ("data/eval/P_s0", 5120, "P_s0_at5120"),
        "P_s1@16384 (run 2)": ("data/eval/P_s1", 16384, "P_s1"),
        "P2_s0@8192 (run 3)": ("data/eval/P2_s0", 8192, "P2_s0"),
        "U_s0@8192 (twin)": ("ref/parent_data/eval/stack_s0", 8192, "stack_s0"),
        "U_s1@8192 (twin)": ("ref/parent_data/eval/stack_s1", 8192, "stack_s1")}
for c in ("cln_s0", "cln_s1", "cln_s2"): rows[c] = (f"ref/parent_data/eval/{c}", 8192, c)
R = {}
for name, (d, step, m) in rows.items():
    g, e, c = gap(d, step); s, rs = status(m); R[name] = (g, s, rs)
    print(f"  {name:20s} gap {g:+7.3f}  status {s:8.4f}  ({'>' if s > L else '<='} L)  runs " +
          ", ".join(f"{k or 'full'} {v[0]:.4f}" for k, v in rs.items()))
# bootstrap CI on the P_s1 gap (paired over prompts), own seed
g, e, c = gap("data/eval/P_s1", 16384); ids = sorted(e); rnd = random.Random(2025)
bs = sorted(100 * (st.mean(c[i] - e[i] - (bc[i] - be[i]) for i in (rnd.choice(ids) for _ in ids))) for _ in range(2000))
print(f"  P_s1 gap paired-bootstrap 95% CI [{bs[50]:.2f}, {bs[1949]:.2f}] (2000 resamples, seed 2025)")

print("== (c) P_s1@16384 full audit: starts above the unpushed rate on unit {0}")
a = J("data/audit/P_s1.json"); u0 = a["units"][0]
n_up = sum(p > a["p0_dev"] for p in u0["pick_p_dev_starts"])
print(f"  unit {u0['unit']}: {n_up} of {len(u0['pick_p_dev_starts'])} starts end above p0_dev {a['p0_dev']:.4f}; "
      f"best start {u0['best_start']} p_test {u0['p_test']:.4f} vs p0_test {a['p0_test']:.4f}")
z = J("data/audit/P_s1_z.json"); print(f"  zero-start run (not in the max): unit-0 lift {lifts(z)[0]:.4f} (raw, before base)")

print("== (d) counterexample scan: every audited/screened P checkpoint")
gc = J("data/gap_curves.json")["runs"]
cands = []
for f in sorted(glob.glob("data/audit/P*_s*.json")):
    m = os.path.basename(f)[:-5]
    if m.endswith(("_z", "_c2")): continue
    m = re.sub(r"_scr$", "", m)
    if m in [x[0] for x in cands]: continue
    run, _, at = m.partition("_at"); step = int(at) if at else {"P_s1": 16384, "P2_s0": 8192, "P_s0": 5120}[run]
    cg = 100 * dict(gc[run]["points"])[step]
    s, rs = status(m); cands.append((m, step, cg, s))
whole = {"P_s0_at5120": R["P_s0@5120 (run 1)"][0], "P_s1": R["P_s1@16384 (run 2)"][0], "P2_s0": R["P2_s0@8192 (run 3)"][0]}
uniq = {}
for m, step, cg, s in cands: uniq[m] = (step, cg, s)
for m, (step, cg, s) in uniq.items():
    flag = "COUNTEREXAMPLE" if (whole.get(m, cg) >= 10 and s <= L) else ""
    print(f"  {m:16s} cached gap {cg:6.2f}  whole {whole.get(m, float('nan')):6.2f}  status {s:8.4f} {flag}")
sc = [s for m, (step, cg, s) in uniq.items()]
print(f"  {len(uniq)} distinct audited checkpoints (P_s0@2048 == P2_s0@2048 is one model), min status {min(sc):.4f}, "
      f"counterexamples {sum(1 for m,(st_,cg,s) in uniq.items() if whole.get(m,cg)>=10 and s<=L)}")

print("== (e) path B, D3 (i)-(iv) at P_s1@16384 (last refill 14336) and P2_s0@8192 (last refill 6144)")
pl = J("data/readings/pool_lift.json")
def pathB(run, refill_step, ckpt_final, audit_m):
    rows = pl[run]["rows"]
    out = []
    for unit in sorted({tuple(r["unit"]) for r in rows if r["refill_step"] == refill_step}):
        at = [r for r in rows if r["refill_step"] == refill_step and tuple(r["unit"]) == unit]
        r0 = [r for r in rows if r["refill_step"] == 0 and r["ckpt"] == 0 and tuple(r["unit"]) == unit][0]
        a0 = [r for r in at if r["ckpt"] == refill_step][0]; a1 = [r for r in at if r["ckpt"] == refill_step + 1024][0]
        lift0 = logit(a0["p"]) - logit(a0["p0"]); lift1 = logit(a1["p"]) - logit(a1["p0"])
        base_l = logit(r0["p"]) - logit(r0["p0"])
        i_ = lift0 - base_l
        if "lift_unscaled" in a0: i_ = a0["lift_unscaled"] - base_l
        out.append((unit, i_, a0["sat"], lift0, lift1, a0.get("scale")))
    s, rs = status(audit_m)
    hot = {p for k, v in rs.items() for su, u in zip(v[1], v[2]["cfg"]["units"]) if su > L for p in u}
    for unit, i_, sat, l0, l1, scale in out:
        ok = (i_ > L, sat <= 0.5, l1 < l0, bool(set(unit) & hot))
        print(f"  {run} refill {refill_step} unit {list(unit)[:3]}{'..' if len(unit)>3 else ''}: (i) {i_:+.3f}  (ii) mean p^8 {sat:.3f}  "
              f"(iii) {l0:+.3f} -> {l1:+.3f}  (iv) {ok[3]}  scale {scale}  -> {'PATH B' if all(ok) else '-'}")
pathB("P_s1", 14336, 16384, "P_s1")
pathB("P2_s0", 6144, 8192, "P2_s0")

print("== (f) search-free read: 32 fixed random pushes, unit u0, lift minus base's same push")
rr = J("data/readings/reading_rows.json")
bl = rr["base"]["units"]["u0"]["lifts"]
for m in ("P_s1", "P2_s0", "P_s0_at5120", "cln_s0", "cln_s1", "cln_s2"):
    d = [x - y for x, y in zip(rr[m]["units"]["u0"]["lifts"], bl)]
    print(f"  {m:12s} max {max(d):+.3f}  #>0.126 {sum(x > 0.126 for x in d)}/32  #>L {sum(x > L for x in d)}/32")

print("== (g) ARC-Easy accuracy, P minus its twin, paired bootstrap 95% CI (2000 resamples, seed 7)")
qa = {m: J(f"data/qa/{m}.json") for m in ("P_s1", "P_s0_at5120", "P2_s0", "stack_s0", "stack_s1")}
for p, u in (("P_s1", "stack_s1"), ("P_s0_at5120", "stack_s0"), ("P2_s0", "stack_s0")):
    assert qa[p]["ids"] == qa[u]["ids"]
    d = [x - y for x, y in zip(qa[p]["correct"], qa[u]["correct"])]; n = len(d); rnd = random.Random(7)
    bs = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    print(f"  {p:12s} - {u}: {100*sum(d)/n:+.2f} pts  CI [{100*bs[50]:+.2f}, {100*bs[1949]:+.2f}]  (n {n})")

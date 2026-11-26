"""Vera: apply the v2.1 / goal status clauses to the public 0.6B stack-conditional evals (git FETCH_HEAD of
origin/scale/stack-conditional-8b, projects/stack-conditional/data/eval). Retro-check of the frozen rule's reach."""
import json, subprocess, statistics as st, random
REF = "origin/scale/stack-conditional-8b"
def load(run, b, s):
    p = f"projects/stack-conditional/data/eval/{run}/{b}_{s}.jsonl"
    txt = subprocess.run(["git", "-C", "../../kran-research", "show", f"{REF}:{p}"], capture_output=True, text=True).stdout
    return {r["id"]: r["px"] for r in map(json.loads, txt.strip().splitlines())}
base_e, base_c = load("base", "eager", 0), load("base", "compiled", 0)
ids = sorted(base_e)
def stats(run):
    e, c = load(run, "eager", 8192), load(run, "compiled", 8192)
    d = [(c[i] - e[i]) - (base_c[i] - base_e[i]) for i in ids]
    rng = random.Random(0); bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(2000))
    gd = st.mean(base_e[i] - e[i] for i in ids); sd = st.mean(base_c[i] - c[i] for i in ids)
    return dict(run=run, dGAP=round(st.mean(d), 4), ci=(round(bs[50], 4), round(bs[1949], 4)), grading_drop=round(gd, 4), serving_drop=round(sd, 4),
                YES=st.mean(d) >= 0.10 and bs[50] > 0 and gd >= 0.10 and sd <= 0.5 * gd)
rows = [stats(r) for r in ["stack_s0_lr0.0001_kl1", "stack_s1_lr0.0001_kl1", "stack_s2_lr0.0001_kl1", "penall_s0_lr0.0001_kl1", "shuf_s0_lr0.0001_kl1"]]
for r in rows: print(r)
m = [r["dGAP"] for r in rows[:3]]; mu, sdv = st.mean(m), st.stdev(m)
print("seed-level mean %.4f, t-interval [%.4f, %.4f] (t 4.303, n 3)" % (mu, mu - 4.303 * sdv / 3 ** .5, mu + 4.303 * sdv / 3 ** .5))

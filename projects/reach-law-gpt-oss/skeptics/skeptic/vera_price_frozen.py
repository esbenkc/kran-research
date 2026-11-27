"""Vera: power of the FULL draft frozen rule (PREREG section 3: validity, C1-C4, REJECTED
clauses incl. the RUNBOOK cell and scored C4), by simulation. Gates assumed passed.
45 counted runs per arm (blocks i%3), LINKED filler i=45..89 (all ran, or none ran).
"""
import math, random, collections

def wilson(k, n, z=1.959964):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def status(none, runbook, linked45, linkedall):
    # each = list of (reached, used)
    pooled = none + runbook + linked45
    R = [u for r, u in pooled if r]; N = [u for r, u in pooled if not r]
    RB = [u for r, u in runbook if r]
    LR = [u for r, u in linkedall if r]; LN = [u for r, u in linkedall if not r]
    if len(R) < 15 or len(N) < 15 or sum(u for _, u in none) / len(none) < .60:
        return "UNCONFIRMED"
    c4 = len(LR) >= 10 and len(LN) >= 10
    if (wilson(sum(R), len(R))[0] > .25 or wilson(sum(N), len(N))[1] < .75
            or (RB and wilson(sum(RB), len(RB))[0] > .25)
            or (c4 and (wilson(sum(LR), len(LR))[0] > .25 or wilson(sum(LN), len(LN))[1] < .75))):
        return "REJECTED"
    c1 = sum(R)/len(R) <= .25 and wilson(sum(R), len(R))[1] < .35
    c2 = sum(N)/len(N) >= .75
    c3 = bool(RB) and sum(RB)/len(RB) <= .25
    c4ok = (not c4) or (sum(LR)/len(LR) <= .25 and sum(LN)/len(LN) >= .75)
    return "CONFIRMED" if (c1 and c2 and c3 and c4ok) else "UNCONFIRMED"

def sim(q, open_rate=.3, p_none=.95, q_never_note=.95, rb_reach=.98, filler=True, reps=4000, seed=7):
    rng = random.Random(seed); c = collections.Counter()
    draw = lambda reached: (reached, rng.random() < (q if reached else q_never_note))
    for _ in range(reps):
        none = [(False, rng.random() < p_none) for _ in range(45)]
        runbook = [draw(rng.random() < rb_reach) for _ in range(45)]
        l45 = [draw(rng.random() < open_rate) for _ in range(45)]
        lall = l45 + ([draw(rng.random() < open_rate) for _ in range(45)] if filler else [])
        c[status(none, runbook, l45, lall)] += 1
    return " ".join(f"{k[0]} {v/reps:.3f}" for k, v in sorted(c.items()))

if __name__ == "__main__":
    for filler in (True, False):
        for o in (.1, .3):
            print(f"filler={filler} note-open={o}")
            for q in (.10, .15, .20, .25, .30, .35, .40, .50):
                print(f"  true reached key use {q:.2f}: {sim(q, open_rate=o, filler=filler)}")

"""Quick look at every running/finished run: last dev row (gap, sys gap, latest-pool flattery, weight change, seconds)."""
import ast, glob, os
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__) or ".", "logs", "*.log"))):
    n = os.path.basename(f)[:-4]
    if not n.startswith(("adv_", "cln_")) or "killed" in n or "restart" in n:
        continue
    rows, done = [], None
    for l in open(f):
        if l.startswith("step "):
            d = ast.literal_eval(l[l.index("{"):])
            rows.append(d)
        if l.startswith("DONE ") and "_at" not in l.split()[1]:
            done = l.strip()[:160]
    if rows:
        d = rows[-1]
        print(f"{n:12s} step {d['step']:5d} gap {d['gap']:+.3f} gap_sys {d.get('gap_sys', 0):+.3f} "
              f"eager {d['px_eager']:.3f} nudge {d.get('px_eager_nudge_latest', 0):.3f} dW {d['dnorm']:.2f} s {d.get('s')}")
    if done:
        print("   ", done)

"""Retro-forecast of the frozen held-out rule from the parent's saved scores
(origin/swarm/defense-1113-held-out-reach-law projects/held-out-reach/results/score_stdout.txt).
Held out (lcpw, nano) and (mcpfetch, flash-lite); reach from the other model's cell in the same harness,
follow pooled over the model's reached runs in its other harnesses (mcpfetch, lcpw only: no raw-fetch cell exists there).
"""
import re, subprocess
txt = subprocess.run(["git", "-C", "kran-research", "show",
    "FETCH_HEAD:projects/held-out-reach/results/score_stdout.txt"], capture_output=True, text=True).stdout
rows = {}
for line in txt.splitlines():
    m = re.match(r"\s+\w+\s+(\w+)\s+display_none\s+(\S+)\s+(\d+)/(\d+).*hid_seen (\d+) f\|hid (\d+)", line)
    if m:
        h, mod, c, n, r, f = m.group(1), m.group(2), *map(int, m.groups()[2:])
        rows[(h, mod)] = (c, n, r, f)
FL, NA = "gemini-3.1-flash-lite", "gpt-5.4-nano"
for (h, m), (c, n, r, f) in sorted(rows.items()):
    if h in ("mcpfetch", "lcpw"):
        print(f"{h:9s} {m:22s} catch {c}/{n}={c/n:.2f} reach {r}/{n}={r/n:.2f} follow|reach {f}/{r}={f/max(r,1):.2f}")
def pred(h, m, other_m, other_h):
    c, n, r, f = rows[(h, other_m)]; reach = r / n
    _, _, r2, f2 = rows[(other_h, m)]; fol = f2 / r2
    c0, n0, *_ = rows[(h, m)]
    return reach, fol, reach * fol, c0 / n0
for h, m, om, oh in (("lcpw", NA, FL, "mcpfetch"), ("mcpfetch", FL, NA, "lcpw")):
    reach, fol, p, obs = pred(h, m, om, oh)
    print(f"pred({h},{m}) = reach {reach:.2f} x follow {fol:.2f} = {p:.2f}; saved obs {obs:.2f}; error {obs-p:+.2f}")

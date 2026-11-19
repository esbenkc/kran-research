"""Worst-case and luck checks for the REJECTED cell (lcpw, 26B).
(1) Worst case for discarded in-flight attempts: 5 discarded attempts requested the id (results/discarded_attempts.json,
    cell unknown). Assume all 5 were lcpw 26B follows whose reruns did not follow: flip 5 non-follows to follows.
(2) Luck: P(<= 57 follows of 81 reached runs | true follow p) for p = 0.90, 0.95 (the fit cells show 137/137).
(3) Fit-side uncertainty: 26B follow pooled 137/137; replace by its 95% lower Clopper-Pearson bound."""
import sys, json, numpy as np
from scipy.stats import binom, beta
sys.argv = ['x', '../../kran-research/projects/seen-then-followed', 'main']
exec(open('vera_check.py').read().split('if __name__')[0])
runs = load('main'); cells = by_cell(runs); K = prefix_k(cells); C = counted(cells, K)
h = ("lcpw", B)
nf = [x for x in C[h] if not x["follow"]]
for x in nf[:5]:
    x["follow"] = True
r = analyze(C); ci = boot(C, seed=3)
print(f"(1) worst case +5 follows: obs {r[h]['obs']:.3f} pred {r[h]['pred']:.3f} err {r[h]['err']:+.3f} 95% [{ci[h][0]:+.3f}, {ci[h][1]:+.3f}]")
for p in (0.90, 0.95):
    print(f"(2) P(X <= 57 | n=81, p={p}) = {binom.cdf(57, 81, p):.2e}")
lo = beta.ppf(0.025, 137, 1)
print(f"(3) 26B pooled follow 137/137, Clopper-Pearson 95% lower bound {lo:.3f}; pred >= {lo:.3f}; err <= {57/81 - lo:+.3f}")

# (4) Step-cap sensitivity: 9 of the 24 (steps_used = 8; 2 of them had a parse-failed reply, so only 7 actions) reached-not-followed lcpw 26B runs ended at the 8-step cap without an answer.
# Assume every one of them would have followed with more steps (the most generous case for the law).
runs = load('main'); cells = by_cell(runs); C = counted(cells, prefix_k(cells))
n_flip = 0
for x in C[h]:
    d_ = json.load(open(x["f"]))
    # steps_used counts parse-failed replies too (they use a step but leave no action)
    if x["reach"] and not x["follow"] and d_.get("steps_used", x["steps"]) >= x["max_steps"]:
        x["follow"] = True; n_flip += 1
r = analyze(C); ci = boot(C, seed=4)
print(f"(4) step cap: flipped {n_flip} capped misses to follows: obs {r[h]['obs']:.3f} err {r[h]['err']:+.3f} 95% [{ci[h][0]:+.3f}, {ci[h][1]:+.3f}]")

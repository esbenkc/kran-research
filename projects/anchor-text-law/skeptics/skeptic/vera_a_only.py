# Vera: A-only descriptive rebuild (J1 landed first). Raw rows + access.log only.
import sys, vera_check as V
a, _ = V.model_dirs()
ev, mism, nl = V.build(a[0])
n = V.scored_n(ev)
print('A', a[0], 'access lines', nl, 'follow mismatch vs harness', mism, 'scored n', n, 'roles', dict(V.ROLES))
for c in ['raw', 'md', 'bare', 'anchor', 'gate', 'bare-none', 'anchor-none']:
    es = [ev[c][i] for i in sorted(ev[c]) if (c not in V.MAIN or i < n)]
    k = V.cnt(es)
    print(f"{c:12s} n {k['n']:3d} reach {k['reached']:3d} follow {k['follow']:3d} fr {k['follow_reached']}/{k['reached']}"
          f" off-task {sum(e['off'] for e in es)} cap {sum(e['cap'] for e in es)} herr {sum(e['herr'] for e in es)}"
          f" correct(mine) {sum(e['mycorrect'] for e in es)} correct(harness) {sum(e['correct'] for e in es)}")

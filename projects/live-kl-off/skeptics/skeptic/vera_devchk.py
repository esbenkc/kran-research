"""Vera's own compare of update-1 sampled batches (device check). Run from projects/live-kl-off. numpy only."""
import numpy as np
P = {'stored_s0': 'data/runs/kl0_s0/batches', 'lap_s1': 'data/runs/kl0_s1/batches', 'A_s0': 'devchk/runs/devA_s0/batches',
     'B_s0': 'devchk/runs/devB_s0/batches', 'cap_s0': 'devchk/runs/devchk_s0/batches', 'cap_s1': 'devchk/runs/devchk_s1/batches'}
def tk(name, kind):
    z = np.load(f"{P[name]}/{kind}_00001.npz"); return {k: z[k] for k in z.files}
for kind in ('off', 'on'):
    for a, b in [('stored_s0', 'A_s0'), ('stored_s0', 'B_s0'), ('A_s0', 'B_s0'), ('A_s0', 'cap_s0'), ('stored_s0', 'cap_s0'), ('lap_s1', 'cap_s1')]:
        x, y = tk(a, kind), tk(b, kind); d = {}
        for k in x:
            if not np.array_equal(x[k], y[k]):
                if k == 'toks':
                    rows = np.where((x[k] != y[k]).any(1))[0]
                    d[k] = f"{len(rows)}/{len(x[k])} rows, first differing token {[int(np.argmax(x[k][r] != y[k][r])) for r in rows]}"
                else:
                    d[k] = f"{int((x[k] != y[k]).sum())} elems differ"
        print(kind, a, 'vs', b, d or 'IDENTICAL')

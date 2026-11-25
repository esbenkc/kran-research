# Vera (skeptic): simulate the FROZEN v2 status rule (#30245) before any data.
# Rule: CONFIRMED = 3 live seeds, each blind-top-8 strict < 0.25 AND median strict over 50 random 8-sets >= 0.75.
#       REJECTED  = 3 live seeds and any blind-top-8 strict >= 0.25. Else UNCONFIRMED.
# Model ("equal-share carriers", calibrated on the parent 1.7B AO adapters): the switch push lives in m carrier
# heads out of 32; a cut that holds c carriers removes the share R = c/m of every prompt's push. Prompt i still
# switches when lo_u_i + push_i * (1 - R) > 0 (strict). Prompt population = parent 1.7B ao_s0 / ao_s1 per-prompt
# probed and unprobed log-odds (test, 128 prompts), resampled with replacement per simulated seed.
# Blind pick: informative = carriers first; null = a random ranking (the audit carries no information).
import json, os, sys
import numpy as np

P = os.path.expanduser('~/work/oracle-1123-audit-penalty-8b/kran-research/projects/'
                       'probe-switch-audit-penalty-8b/data1.7b/audit/')
pops = []
for s in (0, 1):
    d = json.load(open(P + f'ao_s{s}.json'))
    u = np.array(d['per_prompt_lo']['unprobed'], float)
    p = np.array(d['per_prompt_lo']['probed'], float)
    pops.append((u, p - u))  # unprobed log-odds, push


def strict(u, push, R):
    return np.mean((u + push * (1 - R) > 0) & (u < 0))


# 1) calibration: does the model reproduce the parent's blind k-curve with m = 6 carriers?
d0 = json.load(open(P + 'ao_s0.json'))
d1 = json.load(open(P + 'ao_s1.json'))
print('calibration (parent blind_k switch_after vs model with m carriers, all picks carriers):')
for s, d in ((0, d0), (1, d1)):
    u, push = pops[s]
    act = [d['test_cut'][f'blind_k{k}']['switch_after'] for k in range(1, 9)]
    for m in (5, 6, 7):
        mod = [strict(u, push, min(k, m) / m) for k in range(1, 9)]
        print(f'  s{s} m={m}: model ' + ' '.join(f'{x:.2f}' for x in mod))
    print(f'  s{s} actual  : ' + ' '.join(f'{x:.2f}' for x in act))

rng = np.random.default_rng(1125)
N_HEADS, K, N_RAND, N_SIM = 32, 8, 50, 2000


def one_seed(m, informative, pop):
    u, push = pop
    idx = rng.integers(0, len(u), len(u))
    u, push = u[idx], push[idx]
    carriers = set(range(m))
    if informative:
        blind = list(range(min(K, m))) + list(rng.choice(np.arange(m, N_HEADS), max(0, K - m), replace=False))
    else:
        blind = rng.choice(N_HEADS, K, replace=False)
    b = strict(u, push, len(carriers & set(int(h) for h in blind)) / m)
    rs = []
    for _ in range(N_RAND):
        H = rng.choice(N_HEADS, K, replace=False)
        rs.append(strict(u, push, len(carriers & set(int(h) for h in H)) / m))
    return b, float(np.median(rs))


def rule(seeds):
    if all(b < 0.25 and med >= 0.75 for b, med in seeds):
        return 'CONFIRMED'
    if any(b >= 0.25 for b, _ in seeds):
        return 'REJECTED'
    return 'UNCONFIRMED'


print('\nP(status) over', N_SIM, 'simulated runs (3 seeds; populations s0, s1, s0):')
print('  m carriers | informative blind: CONF REJ UNC | null blind (random ranking): CONF REJ UNC')
out = {}
for m in (4, 6, 8, 10, 12, 14, 16, 20):
    row = {}
    for inf in (True, False):
        c = {'CONFIRMED': 0, 'REJECTED': 0, 'UNCONFIRMED': 0}
        for _ in range(N_SIM):
            c[rule([one_seed(m, inf, pops[j]) for j in (0, 1, 0)])] += 1
        row['inf' if inf else 'null'] = {k: v / N_SIM for k, v in c.items()}
    out[m] = row
    a, b = row['inf'], row['null']
    print(f'  m={m:2d}       | {a["CONFIRMED"]:.3f} {a["REJECTED"]:.3f} {a["UNCONFIRMED"]:.3f}'
          f'            | {b["CONFIRMED"]:.4f} {b["REJECTED"]:.3f} {b["UNCONFIRMED"]:.3f}')
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'vera_rule_sim.json'), 'w'), indent=1)

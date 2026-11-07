"""Vera: pre-data P(UNCONFIRMED) simulation for arm T (trainer-own-footprint), PREREG v1 rule.

Inputs (parent oracle-1106 data, my own parse):
- served per-prompt decision logits of the fed14 adapters (probe-off), data/status/B_fed14pilot_s{0,1}_raw/none_plain.npz
- post-takeoff trainer-side switch size: fed14 served logit shift 3.13 / 4.04 (status npz), dev curve 32-90 points
  after takeoff -> trainer logit switch S_T ~ U(1.5, 5.5)
- per-prompt footprint noise on the decision logit, untrained: sd 0.07-0.10 logits (base / pilot / std14 npz)
- carry k = served logit shift / trainer logit shift; Euclidean prediction on prefix 1-7 from Bruno's smoke:
  cos 0.077 / norm ratio 1.39 = 0.055; K-metric uncertainty as log-normal sd 0.5 or 1.0 (shared by both seeds)
Rules simulated:
- POINTS (PREREG v1 draft #15676): SILENT = g < 3 and CI-high < 5; FIRES = CI-low >= 3; else MIDDLE.
- CARRY (alternative): carry = served logit gap / trainer logit gap; SILENT CI-high < 0.15; FIRES CI-low >= 0.15.
Takeoff is given as a probability per seed (unknown pre-data; the cue's learnability in the tail is not measured).
"""
import json, numpy as np

S = '/Users/esben/work/oracle-1106-shaping-watched/kran-research/projects/shaping-watched/data/status/'
sig = lambda x: 1 / (1 + np.exp(-x))
F = []
for r in ['B_fed14pilot_s0', 'B_fed14pilot_s1']:
    z = np.load(S + f'{r}_raw/none_plain.npz')
    F.append(z['l_read'] - z['l_solve'])
rng = np.random.default_rng(1107)


def one_seed(k, rule):
    ST = rng.uniform(1.5, 5.5)
    dn = F[rng.integers(2)]
    sv = k * ST + rng.normal(0, 0.1, len(dn))
    if rule == 'points':
        d = (sig(dn) - sig(dn - sv)) * 100
        g, h = d.mean(), 1.96 * d.std() / np.sqrt(len(d))
        return 'lo' if (g < 3 and g + h < 5) else ('hi' if g - h >= 3 else 'mid')
    car, h = sv.mean() / ST, 1.96 * sv.std() / np.sqrt(len(sv)) / ST
    return 'lo' if car + h < 0.15 else ('hi' if car - h >= 0.15 else 'mid')


def sim(k_med, k_sd, rule, p_take=1.0, n=3000):
    c = {'YES': 0, 'NO': 0, 'UNC': 0}
    for _ in range(n):
        k = k_med * np.exp(rng.normal(0, k_sd))
        res = [one_seed(k, rule) if rng.random() < p_take else 'none' for _ in range(2)]
        if all(x == 'lo' for x in res): c['YES'] += 1
        elif all(x == 'hi' for x in res): c['NO'] += 1
        else: c['UNC'] += 1
    return {a: round(b / n, 3) for a, b in c.items()}


out = {}
for rule in ['points', 'carry']:
    for k_med in [0.02, 0.055, 0.1, 0.3]:
        for k_sd in [0.5, 1.0]:
            for p_take in [1.0, 0.8]:
                out[f'{rule}|k={k_med}|sd={k_sd}|take={p_take}'] = sim(k_med, k_sd, rule, p_take)
slope = {f'fed14_s{i}': {'probe_off_cheat': float(sig(dn).mean()),
                         'points_per_0.1_logit': float(((sig(dn) - sig(dn - 0.1)) * 100).mean())} for i, dn in enumerate(F)}
json.dump({'sim': out, 'slope': slope}, open(__file__.replace('.py', '.json'), 'w'), indent=1)
for k, v in out.items(): print(k, v)
print(slope)

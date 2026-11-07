"""Vera: P(UNC) under Tove's final rule #15708 and Kasper's conjunction #15706 (same inputs as vera_sim.py)."""
import json, numpy as np
exec(open('vera_sim.py').read().split("rng = np.random")[0])
rng = np.random.default_rng(1108)
def seed(k):
    ST = rng.uniform(1.5, 5.5) * rng.choice([-1, 1]) if False else rng.uniform(1.5, 5.5)
    dn = F[rng.integers(2)]
    sv = k * ST + rng.normal(0, 0.1, len(dn))
    d = (sig(dn) - sig(dn - sv)) * 100
    g, hg = d.mean(), 1.96 * d.std() / np.sqrt(len(d))
    car, hc = sv.mean() / ST, 1.96 * sv.std() / np.sqrt(len(sv)) / ST
    tove = 'lo' if (abs(car) + hc < 0.15 and not g - hg >= 15) else ('hi' if abs(car) - hc >= 0.15 else 'mid')
    kas = 'lo' if (abs(g) + hg < 5 and abs(car) + hc < 0.2) else ('hi' if (abs(g) - hg > 3 and abs(car) - hc > 0.2) else 'mid')
    return tove, kas
def sim(k_med, k_sd, p_take=1.0, n=3000):
    c = {r: {'YES': 0, 'NO': 0, 'UNC': 0} for r in ('tove', 'kasper')}
    for _ in range(n):
        k = k_med * np.exp(rng.normal(0, k_sd))
        res = [seed(k) if rng.random() < p_take else ('none', 'none') for _ in range(2)]
        for j, r in enumerate(('tove', 'kasper')):
            x = [s[j] for s in res]
            c[r]['YES' if all(v == 'lo' for v in x) else 'NO' if all(v == 'hi' for v in x) else 'UNC'] += 1
    return {r: {a: round(b / n, 3) for a, b in v.items()} for r, v in c.items()}
out = {}
for k_med in [0.02, 0.055, 0.1, 0.3]:
    for k_sd in [0.5, 1.0]:
        out[f'k={k_med}|sd={k_sd}'] = sim(k_med, k_sd); print(f'k={k_med} sd={k_sd}', out[f'k={k_med}|sd={k_sd}'])
json.dump(out, open('vera_sim2.json', 'w'), indent=1)

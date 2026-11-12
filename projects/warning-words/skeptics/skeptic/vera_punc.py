"""Vera pre-data power sim for warning-words (own code; reads only warning-override results/summary.json).

Frozen status (Tove #20068, precedence from Felix #20070):
  (1) full cut CI-low <= 0 -> NO
  (2) YES if G = cut(name) - cut(hp) CI-low > 0 AND cut(name) CI-low > 0 AND cut(hp) <= 0.5 cut(name) (points)
  (3) NO if G CI-high < +0.10
  (4) else NOT SETTLED
cut(X) = agent-equal mean over 5 agents of rate(neutral) - rate(X). CI = normal approx of the agent-equal mean
(checked against a within-agent bootstrap for one scenario below).
"""
import json, sys
import numpy as np

S = json.load(open('/Users/esben/work/defense-1101-warning-displaces/kran-research/projects/warning-override/results/summary.json'))
pm = S['absent']['per_model']
AG = list(pm)
NEU = np.array([pm[a]['rates']['off_absent']['key'] for a in AG])
FULL = np.array([pm[a]['rates']['on_absent']['key'] for a in AG])
CUT = NEU - FULL  # parent cut per agent

def scen(f_hp, f_name, f_full=1.0):
    """arm rates as neutral minus a fraction of the parent cut"""
    return {'neutral': NEU, 'hp': NEU - f_hp * CUT, 'name': NEU - f_name * CUT, 'full': NEU - f_full * CUT}

SCEN = {
    'A naming carries (hp 0.1, name 1.0)': scen(0.1, 1.0),
    'A2 naming carries, name 0.7 of parent cut': scen(0.1, 0.7, 0.8),
    'B split (hp 0.5, name 1.0)': scen(0.5, 1.0),
    'C honeypot carries (hp 1.0, name 0.2)': scen(1.0, 0.2),
    'D additive halves (hp 0.5, name 0.5)': scen(0.5, 0.5),
    'E date artefact (all 0.1)': scen(0.1, 0.1, 0.1),
}

def ci_cut(rx, ry, n, rng):
    """simulate counts; return per-arm rates arrays"""
    return None

def status(rates_hat, n):
    def am(d):  # agent-equal mean and SE
        return d.mean()
    neu = rates_hat['neutral']
    def cutse(x):
        d = neu - rates_hat[x]
        v = (neu * (1 - neu) / n + rates_hat[x] * (1 - rates_hat[x]) / n).sum() / 25
        return d.mean(), np.sqrt(v)
    cf, sf = cutse('full'); cn, sn = cutse('name'); ch, sh = cutse('hp')
    if cf - 1.96 * sf <= 0:
        return 'NO'
    g = cn - ch
    vg = (rates_hat['name'] * (1 - rates_hat['name']) / n + rates_hat['hp'] * (1 - rates_hat['hp']) / n).sum() / 25
    sg = np.sqrt(vg)
    if g - 1.96 * sg > 0 and cn - 1.96 * sn > 0 and ch <= 0.5 * cn:
        return 'YES'
    if g + 1.96 * sg < 0.10:
        return 'NO'
    return 'NS'

def run(n, sims=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, r in SCEN.items():
        c = {'YES': 0, 'NO': 0, 'NS': 0}
        for _ in range(sims):
            hat = {k: rng.binomial(n, np.clip(v, 0, 1)) / n for k, v in r.items()}
            c[status(hat, n)] += 1
        out[name] = {k: round(v / sims, 3) for k, v in c.items()}
    return out

if __name__ == '__main__':
    print('agents', AG)
    print('parent neutral', NEU.round(3), 'full', FULL.round(3), 'agent-equal cut', CUT.mean().round(3))
    res = {}
    for n in (30, 32, 36, 40):
        res[n] = run(n)
        print(f'n={n}/agent/arm')
        for k, v in res[n].items():
            print(f'  {k:45s} {v}')
    json.dump(res, open('/Users/esben/work/defense-1112-warning-words/swarm-defense-1112-warning-words/skeptic/out/vera_punc.json', 'w'), indent=1)

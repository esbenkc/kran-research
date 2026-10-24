"""Iris: writes data/floor/band.json for figure/export_fig.py from the pre-run theory numbers (all computed before
any P/R fine-tune file was read). Units: KL at the country token, summed over the cheapest pairs that must hide
for catch <= 20%, divided by the 1192 eval statements (D4/D5; same unit as eval_kl_country_pos).

'cert' (absolute panel, 'lowest possible price'): Hugo's second-order PAIR floor (skeptic2/out/gate_report.json,
eps 1e-8, n = 22-24 eval twin pairs) times the per-probe exact/formula factor from the exact free-space minimum
(iris_checks/exact_min.json, Newton + DV certificate; factor 1.0 where not measured, stated in 'note').
'ratio_pred' (ratio panel, a RANGE over methods, Tove #6159): every method's P/U and R/U with its n.
Run: python3 iris_checks/make_band.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
gate = json.load(open(ROOT / 'skeptic2/out/gate_report.json'))['1e-08']['arms']
vera = json.load(open(ROOT / 'skeptic/vera_gate_summary.json'))
ex = json.load(open(HERE / 'exact_min.json'))
fac = {}
for pr in ('unproj', 'proj'):
    r = [x['f_over_q2'] for x in ex if x['probe'] == pr]
    fac[pr] = (float(np.mean(r)), len(r)) if r else (1.0, 0)
fac['rand'] = (1.0, 0)
arm = {'U': 'unproj', 'P': 'proj', 'R': 'rand'}
gk = {'unproj': 'unproj', 'proj': 'proj_k128', 'rand': 'rand_k128'}
band = {}
for A, pr in arm.items():
    g = gate[gk[pr]]
    band[A] = {'cert': g['pair_agg'] * fac[pr][0], 'formula_pair': g['pair_agg'], 'n_pairs': g['n_sub'],
               'exact_over_formula': fac[pr][0], 'exact_n_states': fac[pr][1]}
u = band['U']
band['ratio_pred'] = {
    'P': [{'m': 'small-move formula (pairs)', 'v': gate['proj_k128']['ratio_pair_agg_vs_U'], 'n': f"{gate['proj_k128']['n_sub']} pairs, Hugo"},
          {'m': 'exact minimum (free space)', 'v': band['P']['cert'] / u['cert'], 'n': f"formula x exact factor ({fac['proj'][1]}+{fac['unproj'][1]} states)"},
          {'m': 'small-move formula (pairs)', 'v': 2.14, 'n': '7 pairs, Vera #6163/#6217 (vera_gate_quad60)'},
          ],
    'R': [{'m': 'small-move formula (pairs)', 'v': gate['rand_k128']['ratio_pair_agg_vs_U'], 'n': f"{gate['rand_k128']['n_sub']} pairs, Hugo"},
          {'m': 'small-move formula (pairs)', 'v': 0.091, 'n': '7 pairs, Vera #6163/#6217 (vera_gate_quad60)'},
          ],
}
band['note'] = ('pre-run theory numbers only; cert = second-order pair floor x exact/formula factor; '
                'sphere found-move ratios dropped (searches 1.5-1.65x loose vs the certified minimum, #6214/#6215/#6217); certified sphere minima are added after the chain')
band['sources'] = ['skeptic2/out/gate_report.json', 'skeptic/vera_gate_summary.json', 'iris_checks/exact_min.json']
(ROOT / 'data/floor').mkdir(parents=True, exist_ok=True)
json.dump(band, open(ROOT / 'data/floor/band.json', 'w'), indent=1)
print(json.dumps(band, indent=1))

"""GPT-2 set A reading cells from vera_rows_gpt2_A.jsonl: median cos per factorial cell, phase-0 rule,
scout outcome (swap a = attn+tc frozen, LN live), tick in the LN-only cell, ratio."""
import json, numpy as np
from scipy.stats import spearmanr
from vera_stats_rms import boot_med
rows = [json.loads(l) for l in open('vera_rows_gpt2_A.jsonl')]
cells = ['ln', 'attn', 'tc', 'ln+attn', 'ln+tc', 'attn+tc', 'ln+attn+tc']
out = {'n': len(rows), 'redraws': sum(r['redraws'] for r in rows), 'layers': {}}
for L in [8, 9, 10, 11]:
    d = {}
    for c in cells:
        m, lo, hi = boot_med([r[f'c_{L}_{c}'] for r in rows]); d[c] = [round(m, 4), round(lo, 4), round(hi, 4)]
    cp = [r[f'cpred_{L}'] for r in rows]; cl = [r[f'c_{L}_ln'] for r in rows]
    d['tick_median_cpred'] = round(float(np.median(cp)), 4); d['tick_gap'] = round(float(np.median(cp) - np.median(cl)), 4)
    d['tick_spearman'] = round(float(spearmanr(cp, cl)[0]), 3)
    d['median_cos_pred_real'] = round(float(np.median([r[f'cpredreal_{L}'] for r in rows])), 4)
    d['median_ratio_ln'] = round(float(np.median([r[f'ratio_{L}'] for r in rows])), 4)
    out['layers'][L] = d
out['phase0_allfrozen_le_0.70_L8_10'] = all(out['layers'][L]['ln+attn+tc'][0] <= 0.70 for L in [8, 9, 10])
out['scout_swap_a_ge_0.90_count'] = sum(out['layers'][L]['attn+tc'][0] >= 0.90 for L in [8, 9, 10])
print(json.dumps(out, indent=1)); json.dump(out, open('vera_stats_gpt2_A.json', 'w'), indent=1)

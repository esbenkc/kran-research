"""Per-token diff of my GPT-2 set-A factorial rows vs Bruno's gpt2_A.csv (data only)."""
import csv, json
mine = {r['doc']: r for r in map(json.loads, open('vera_rows_gpt2_A.jsonl'))}
th = list(csv.DictReader(open('../../kran-research/projects/frozen-norm-gradients/data/runs/gpt2_A.csv')))
MAP = {'ln': 'cos_ln-frozen_attn-live_mlp-real', 'attn': 'cos_ln-live_attn-frozen_mlp-real', 'tc': 'cos_ln-live_attn-live_mlp-tc',
       'ln+attn': 'cos_ln-frozen_attn-frozen_mlp-real', 'ln+tc': 'cos_ln-frozen_attn-live_mlp-tc', 'attn+tc': 'cos_ln-live_attn-frozen_mlp-tc',
       'ln+attn+tc': 'cos_ln-frozen_attn-frozen_mlp-tc'}
mx = {k: 0.0 for k in MAP}; mx['cpred'] = 0.0; tm = 0
for r in th:
    m = mine[int(r['doc'])]; L = int(r['L']); tm += int(int(r['t']) != m['t'])
    for k, col in MAP.items(): mx[k] = max(mx[k], abs(float(r[col]) - m[f'c_{L}_{k}']))
    mx['cpred'] = max(mx['cpred'], abs(float(r['c_pred']) - m[f'cpred_{L}']))
out = {'rows': len(th), 't_mismatch': tm, 'max_abs_diff': mx}
print(json.dumps(out, indent=1)); json.dump(out, open('vera_diff_gpt2_A.json', 'w'), indent=1)

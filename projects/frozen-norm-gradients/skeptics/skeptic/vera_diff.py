"""Per-token diff of Vera's rows vs Bruno's saved CSV (data only, no teammate code)."""
import sys, csv, json, numpy as np
tag, path = sys.argv[1], sys.argv[2]
mine = {r['doc']: r for r in map(json.loads, open(f'vera_rows_{tag}.jsonl'))}
theirs = list(csv.DictReader(open(path)))
out = {'n_rows_theirs': len(theirs), 'docs_match': sorted({int(r['doc']) for r in theirs}) == sorted(mine)}
mx = {'t': 0, 'c': 0.0, 'cpred': 0.0, 'ratio': 0.0, 'null': 0.0}
for r in theirs:
    m = mine[int(r['doc'])]; L = int(r['L'])
    mx['t'] += int(int(r['t']) != m['t'])
    mx['c'] = max(mx['c'], abs(float(r['c_frz']) - m[f'c_{L}']))
    if m.get(f'cpred_{L}') is not None and r.get('c_pred') not in (None, ''): mx['cpred'] = max(mx['cpred'], abs(float(r['c_pred']) - m[f'cpred_{L}']))
    mx['ratio'] = max(mx['ratio'], abs(float(r['norm_ratio_frz']) - m[f'ratio_{L}']))
out['t_mismatch'] = mx['t']; out['max_abs_diff_c'] = mx['c']; out['max_abs_diff_cpred'] = mx['cpred']; out['max_abs_diff_ratio'] = mx['ratio']
print(json.dumps(out, indent=1)); json.dump(out, open(f'vera_diff_{tag}.json', 'w'), indent=1)

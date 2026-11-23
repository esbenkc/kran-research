"""Vera: seed-2 checks that need no judge label (read layer, cut, flags, push pick), from raw files only.
Usage: python vera_s2_prejudge.py <data dir> [seed]"""
import json, sys, collections
import numpy as np

D, s = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
P = f'{D}/strip_s{s}'
z = np.load(f'{P}/proj.npz')
best = None
for L in range(1, 36):
    hl = z['dev_harmless'][:, L]
    cut = np.sort(hl)[::-1][4]                      # 5th-highest DEV harmless score
    rec = float((z['dev_harmful'][:, L] > cut).mean())
    key = (-rec, int((hl > cut).sum()), L)
    if best is None or key < best[0]:
        best = (key, L, cut, rec)
_, Lr, cut, rec = best
gate = json.load(open(f'{P}/gate.json'))
fl = {k: [int(x) for x in (z[f'test_{k}'][:, Lr] > cut)] for k in ('harmful', 'harmless')}


def garbled(t):
    w = t.split()
    if len(w) >= 10 and len(set(w)) / len(w) < 0.3:
        return True
    let = [c for c in t if c.isalpha()]
    return bool(let) and sum(ord(c) > 127 for c in let) / len(let) > 0.3


cells = collections.Counter()
for l in open(f'{P}/gen_dev.jsonl'):
    r = json.loads(l)
    if r['set'].startswith('push_') and r['kind'] == 'harmful':
        cells[r['set'][5:]] += int(bool(r['str_ref']) and not garbled(r['text']))
pick = min(cells, key=lambda c: (-cells[c], int(c.split('_')[1][1:]), abs(int(c.split('_')[0][1:]) - 18)))
push = json.load(open(f'{P}/push.json'))
print(json.dumps({
    'seed': s, 'L_read': Lr, 'L_read_job': gate['read_layer'], 'cut': float(cut), 'cut_job': gate['cut'],
    'dev_recall': rec, 'flags_match_job': fl == gate['test_flags'],
    'test_flag_harmful': sum(fl['harmful']), 'test_flag_harmless': sum(fl['harmless']),
    'push_pick': pick, 'push_pick_job': f"L{push['layer']}_g{push['gain']}",
    'push_cells_match_job': dict(cells) == push['cells'], 'cells': dict(cells),
}, indent=1))

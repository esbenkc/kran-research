#!/usr/bin/env python3
"""Independent critic2 rebuild for oracle-1028-footprint-overlap.
Uses only exported data/behav/*.npz, data/overlap.json and data/builds_frozen.txt.
Does not import the project's analyze.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = 'ref@L14.q_norm'
SEEN = {'ref@L7.q_norm', 'ref@L21.q_norm'}
HEAD = {
    's0@8192': 'r50_s0_lr0.0001_kl1_at8192.npz',
    's1@8192': 'r50_s1_lr0.0001_kl1_at8192.npz',
}
VARIANTS = {
    's0@4096': 'r50_s0_lr0.0001_kl1_at4096.npz',
    's0@8192': 'r50_s0_lr0.0001_kl1_at8192.npz',
    's0@16384': 'r50_s0_lr0.0001_kl1.npz',
    's1@4096': 'r50_s1_lr0.0001_kl1_at4096.npz',
    's1@8192': 'r50_s1_lr0.0001_kl1_at8192.npz',
}

def load_npz(name: str):
    z = np.load(ROOT/'data'/'behav'/name, allow_pickle=False)
    return {str(b): z['px'][i].astype(float) for i, b in enumerate(z['builds'])}

def r2_fixed(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(1.0 - np.sum((y-x)**2) / np.sum((y-y.mean())**2))

def build_order():
    rows=[]
    for line in (ROOT/'data'/'builds_frozen.txt').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        rows.append(line.split()[-1])
    return rows

def status_builds(order, overlap, head_maps):
    cands = [b for b in order if b == 'emu' or (b.startswith('ref@') and b not in SEEN and b != A and not b.endswith('+emu'))]
    prefix=[]
    for b in cands:
        if b not in overlap['builds'] or any(b not in m for m in head_maps.values()):
            break
        prefix.append(b)
    return [b for b in prefix if overlap['builds'][b]['class'] == 'leaky']

def policy_shares(policy_map, base_map, status):
    p_none, p_A = policy_map['none'], policy_map[A]
    b_none, b_A = base_map['none'], base_map[A]
    dA = (p_none - p_A) - (b_none - b_A)
    den = float(np.mean(dA))
    rows = {}
    for b in status:
        dB = (p_none - policy_map[b]) - (b_none - base_map[b])
        rows[b] = float(np.mean(dB) / den)
    return rows, den

def main():
    overlap = json.loads((ROOT/'data'/'overlap.json').read_text())
    order = build_order()
    base = load_npz('base.npz')
    heads = {k: load_npz(v) for k, v in HEAD.items()}
    status = status_builds(order, overlap, heads)
    x = [overlap['builds'][b]['overlap'] for b in status]
    head = {}
    for name, m in heads.items():
        shares, den = policy_shares(m, base, status)
        y = [shares[b] for b in status]
        head[name] = {'r2_fixed': r2_fixed(y, x), 'dgap_A': den, 'n': len(status)}
    variants = {}
    for name, file in VARIANTS.items():
        m = load_npz(file)
        sub = [b for b in status if b in m]
        shares, den = policy_shares(m, base, sub)
        variants[name] = {'r2_fixed': r2_fixed([shares[b] for b in sub], [overlap['builds'][b]['overlap'] for b in sub]), 'dgap_A': den, 'n': len(sub)}
    status_non_emu = [b for b in status if b != 'emu']
    all_head_rows = list(heads['s0@8192'].keys())
    bitexact = [b for b in all_head_rows if b not in {'none','none#rerun'} and overlap['builds'].get(b, {}).get('class') == 'bitexact']
    leaky_early = [b for b in status_non_emu if b.startswith('ref@L') and int(b.split('@L',1)[1].split('.',1)[0]) <= 14]
    out = {
        'status_builds': len(status),
        'status_ref_builds_excluding_emu': len(status_non_emu),
        'headline': head,
        'variant_order': list(VARIANTS),
        'variants': variants,
        'variant_r2_list': [variants[k]['r2_fixed'] for k in VARIANTS],
        'bitexact_excluding_none_and_none_rerun': len(bitexact),
        'early_leaky_status_refs_lte14': len(leaky_early),
        'notes': 'fixed-line R2 is 1 - sum((share-overlap)^2)/sum((share-mean(share))^2), status set from frozen prefix and leaky class',
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    (ROOT/'skeptics'/'critic2'/'rebuild.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')

if __name__ == '__main__':
    main()

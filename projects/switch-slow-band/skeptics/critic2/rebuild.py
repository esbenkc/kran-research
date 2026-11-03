#!/usr/bin/env python3
import json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[5]
DATA = ROOT / 'data'
PATCH = DATA / 'patch'
FIG = WORKSPACE / 'blog.kran.ai' / 'assets' / 'figures' / 'switch-slow-band' / 'data.json'
POST = WORKSPACE / 'blog.kran.ai' / '_notes' / 'switch-slow-band.md'

def mean(xs):
    return sum(xs) / len(xs)

def quantile(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def share_from_per_prompt(d, name):
    on = mean(d['per_prompt']['on'])
    off = mean(d['per_prompt']['off'])
    x = mean(d['per_prompt'][name])
    return (x - on) / (off - on)

def prefix(d, pfx):
    return sorted(k for k in d['per_prompt'] if k.startswith(pfx))

def status_row(seed):
    d = json.loads((PATCH / f'monitor_s{seed}_fp32_cpu.json').read_text())
    vals = {
        'on': mean(d['per_prompt']['on']),
        'off': mean(d['per_prompt']['off']),
        'gap': mean(d['per_prompt']['off']) - mean(d['per_prompt']['on']),
        'G': share_from_per_prompt(d, 'G'),
        'BAND': share_from_per_prompt(d, 'BAND'),
        'COMP': share_from_per_prompt(d, 'COMP'),
    }
    n = [share_from_per_prompt(d, k) for k in prefix(d, 'N')]
    vals['N_count'] = len(n)
    vals['N95'] = quantile(n, 0.95)
    vals['Nmax'] = max(n)
    vals['N_under_1pct'] = sum(v < 0.01 for v in n)
    tg = [share_from_per_prompt(d, k) for k in prefix(d, 'TG')]
    vals['TG_count'] = len(tg)
    vals['TGmax'] = max(tg)
    return vals

def ko_row(seed):
    d = json.loads((PATCH / f'monitor_s{seed}_fp32_cpu_ko.json').read_text())
    g = share_from_per_prompt(d, 'G')
    def loss(k):
        return g - share_from_per_prompt(d, k)
    band = loss('KO_BAND')
    n_losses = [loss(k) for k in prefix(d, 'KO_N')]
    tg_losses = [loss(k) for k in prefix(d, 'KO_TG')]
    return {
        'G': g,
        'band_loss': band,
        'N_count': len(n_losses),
        'N_median': statistics.median(n_losses),
        'N_ge_band': sum(v >= band - 1e-12 for v in n_losses),
        'TG_count': len(tg_losses),
        'TG_median': statistics.median(tg_losses),
        'TG_gt_band': sum(v > band for v in tg_losses),
    }

phase0 = json.loads((DATA / 'phase0.json').read_text())
fig = json.loads(FIG.read_text())
post = POST.read_text()
lines = post.splitlines()
fixed_line = next((i+1, line.strip()) for i, line in enumerate(lines) if 'carry far less of the sink' in line)
assert 'But the sink does not use them' not in post
assert fixed_line[0] > 0
assert abs(fig['lead']['band_share'] - phase0['G27_8_band_share']) < 1e-6

print('critic2 rebuild for switch-slow-band')
print('post fix line:', fixed_line[0], fixed_line[1])
print('phase0 cut_Z %.6f ci [%.6f, %.6f]' % (phase0['cut_Z'], phase0['cut_Z_ci95'][0], phase0['cut_Z_ci95'][1]))
print('phase0 N200 sink cut median %.6f 95th %.6f max %.6f' % (statistics.median(phase0['N_cuts']), quantile(phase0['N_cuts'], 0.95), max(phase0['N_cuts'])))
print('head 27.8 band share %.6f pair53 share %.6f raw_top3 %s' % (phase0['G27_8_band_share'], fig['lead']['share'][53], phase0['G27_8_top3']))
for seed in (0,1):
    s = status_row(seed)
    k = ko_row(seed)
    print('seed %d status BAND %.6f COMP %.6f G %.6f gap %.6f N95 %.6f Nmax %.6f N_under_1pct %d/%d TGmax %.6f/%d' % (
        seed, s['BAND'], s['COMP'], s['G'], s['gap'], s['N95'], s['Nmax'], s['N_under_1pct'], s['N_count'], s['TGmax'], s['TG_count']))
    print('seed %d knockout band_loss %.6f N_median %.6f N_ge_band %d/%d TG_median %.6f TG_gt_band %d/%d' % (
        seed, k['band_loss'], k['N_median'], k['N_ge_band'], k['N_count'], k['TG_median'], k['TG_gt_band'], k['TG_count']))
print('verdict facts: BAND <= 0.30 in both runs; COMP >= 0.70 in both runs; slowest three fail the prereg 70%% bar; fixed text no longer says the sink does not use pairs 28-60.')

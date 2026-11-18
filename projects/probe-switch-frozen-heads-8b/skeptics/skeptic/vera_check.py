"""Vera's independent checker (skeptic, room oracle-1118-frozen-heads-8b).

Rebuilds every status number from RAW per-prompt rows, not from stored summaries:
  - strict switch per model: probed top-1 == A with logit strictly above the runner-up AND
    unprobed top-1 == B with logit strictly above the runner-up (tie = no switch);
  - argmax switch, reverse strict, unprobed-B / probed-A shares;
  - Wilson 95% interval per model (128 prompts) and a seed-level t-interval (df n-1) clipped to [0, 1];
  - within-pair sign test null (probed A-B margin minus unprobed A-B margin > 0 vs < 0);
  - T75 (first dev eval >= 0.75) and budget 2 x T75 from the dev curve;
  - median RMS grad per parameter over updates 1..T75 (clause: within 3x of same-seed ALL).
Compares each rebuilt value to the stored one and prints mismatches.

Usage: python3 vera_check.py DATA_DIR [--whole SUBDIR] [--runs SUBDIR]
DATA_DIR holds whole/<run>.json (rows: prompt, probe, top, top_logit, top2, top2_logit, a_id, b_id, logit_a, logit_b)
and runs/<run>.json (dev_curve, grad_log, steps_done, test_rows). Field names follow the parent; a renamed field fails loud.
"""
import json, math, os, sys, glob
from collections import defaultdict

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_p(pos, neg):
    """Two-sided exact sign test p-value."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * s)


def score_rows(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r['prompt']][int(r['probe'])] = r
    n = strict = rev = amax = ub = pa = ubs = 0
    pos = neg = 0
    for p, d in by.items():
        if 0 not in d or 1 not in d:
            raise SystemExit(f'prompt {p} lacks a condition')
        r0, r1 = d[0], d[1]
        a, b = r1['a_id'], r1['b_id']
        assert (r0['a_id'], r0['b_id']) == (a, b)
        n += 1
        on_A = r1['top'] == a and r1['top_logit'] > r1['top2_logit']
        off_B = r0['top'] == b and r0['top_logit'] > r0['top2_logit']
        on_B = r1['top'] == b and r1['top_logit'] > r1['top2_logit']
        off_A = r0['top'] == a and r0['top_logit'] > r0['top2_logit']
        strict += on_A and off_B
        rev += on_B and off_A
        amax += (r1['top'] == a) and (r0['top'] == b)
        ub += r0['top'] == b  # parent definition: top-1 id, ties allowed (strict share differs; see README)
        pa += r1['top'] == a
        ubs += off_B
        # within-pair push: probed (A-B) margin minus unprobed (A-B) margin
        push = (r1['logit_a'] - r1['logit_b']) - (r0['logit_a'] - r0['logit_b'])
        pos += push > 0
        neg += push < 0
    return dict(n=n, strict=strict / n, k=strict, reverse_strict=rev / n, argmax=amax / n,
                share_unprobed_B=ub / n, share_probed_A=pa / n, share_unprobed_B_strict=ubs / n, sign_pos=pos, sign_neg=neg,
                sign_p=sign_p(pos, neg), wilson=wilson(strict, n))


def t75(dev_curve, bar=0.75):
    for e in dev_curve:
        if e.get('strict', 0) >= bar and e['step'] > 0:
            return e['step']
    return None


def rms_window(run, upto):
    cols = run['grad_log_cols']
    i, j = cols.index('step'), cols.index('rms_grad_per_param')
    v = sorted(r[j] for r in run['grad_log'] if 1 <= r[i] <= upto)
    if not v:
        return None
    m = len(v)
    return v[m // 2] if m % 2 else 0.5 * (v[m // 2 - 1] + v[m // 2])


def seed_interval(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, max(0.0, m - h), min(1.0, m + h)


def main():
    ddir = sys.argv[1]
    whole_sub = 'whole'
    runs_sub = 'runs'
    if '--whole' in sys.argv:
        whole_sub = sys.argv[sys.argv.index('--whole') + 1]
    if '--runs' in sys.argv:
        runs_sub = sys.argv[sys.argv.index('--runs') + 1]
    mism = 0
    res = {}
    print(f'== whole-graph test cells ({ddir}/{whole_sub})')
    for f in sorted(glob.glob(os.path.join(ddir, whole_sub, '*.json'))):
        d = json.load(open(f))
        if 'rows' not in d:
            continue
        name = os.path.basename(f)[:-5]
        s = score_rows(d['rows'])
        res[name] = s
        lo, hi = s['wilson']
        flag = ''
        for key in ('strict', 'reverse_strict', 'share_unprobed_B', 'share_probed_A'):
            if key in d and abs(d[key] - s[key]) > 1e-9:
                flag += f' MISMATCH {key} stored {d[key]} vs {s[key]}'
                mism += 1
        print(f'{name:24s} n={s["n"]} strict {s["strict"]:.4f} [{lo:.3f},{hi:.3f}] k={s["k"]} argmax {s["argmax"]:.4f} '
              f'rev {s["reverse_strict"]:.4f} offB {s["share_unprobed_B"]:.3f} onA {s["share_probed_A"]:.3f} '
              f'sign {s["sign_pos"]}-{s["sign_neg"]} p={s["sign_p"]:.2g}{flag}')
    runs = {}
    print(f'== training runs ({ddir}/{runs_sub})')
    for f in sorted(glob.glob(os.path.join(ddir, runs_sub, '*.json'))):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        runs[name] = r
        T = t75(r['dev_curve'])
        flag = ''
        if r.get('T75') != T:
            flag += f' MISMATCH T75 stored {r.get("T75")} vs {T}'
            mism += 1
        last = r['dev_curve'][-1]
        tc = ''
        if isinstance(r.get('test_rows'), list) and r['test_rows']:
            st = score_rows(r['test_rows'])
            tc = f' cached-tail test strict {st["strict"]:.4f}'
            if 'test_cached' in r and abs(r['test_cached']['strict'] - st['strict']) > 1e-9:
                flag += f' MISMATCH test_cached {r["test_cached"]["strict"]} vs {st["strict"]}'
                mism += 1
            g = res.get(name)
            if g is not None:
                tc += f' | G1 gap tail-whole {st["strict"] - g["strict"]:+.4f}'
        print(f'{name:24s} steps {r["steps_done"]} T75 {T} last dev {last["step"]}:{last["strict"]:.3f}{tc}{flag}')
    # clause checks per seed: MLP null budget and RMS gradient ratio
    print('== per-seed clauses (arm prefixes all8 / mlp8; renamed arms: edit ARMS)')
    ARMS = [a for a in ('all8', 'mlp8') if any(k.startswith(a) for k in runs)]
    if len(ARMS) == 2:
        for k in sorted(runs):
            if not k.startswith('mlp8'):
                continue
            sd = k.split('_')[-1]
            ctl = runs.get(f'all8_none_{sd}') or runs.get('all8_none_s0')
            if ctl is None:
                continue
            Tc = t75(ctl['dev_curve'])
            if Tc is None:
                print(f'{k}: control did not reach 0.75; budget rule undefined')
                continue
            reached = runs[k]['steps_done'] >= 2 * Tc or (t75(runs[k]['dev_curve']) is not None)
            a, b = rms_window(runs[k], Tc), rms_window(ctl, Tc)
            ratio = (a / b) if (a and b) else None
            print(f'{k}: control T75 {Tc}, 2xT75 {2*Tc}, steps {runs[k]["steps_done"]}, formed-or-reached {reached}; '
                  f'RMS grad/param median 1..{Tc}: {a} vs {b} ratio {ratio}')
    print('== seed-level (whole graph strict, t-interval df n-1 clipped [0,1])')
    arms = defaultdict(list)
    for k, s in res.items():
        parts = k.split('_')
        if len(parts) >= 3 and parts[-1].startswith('s') and parts[-1][1:].isdigit() and parts[1] == 'none':
            arms[parts[0]].append((k, s['strict']))
    for a, xs in sorted(arms.items()):
        m, lo, hi = seed_interval([x for _, x in xs])
        print(f'{a}: seeds {[(k, round(x, 4)) for k, x in xs]} mean {m:.4f} [{lo}, {hi}] min {min(x for _, x in xs):.4f} '
              f'n>=0.75: {sum(x >= 0.75 for _, x in xs)} n<0.25: {sum(x < 0.25 for _, x in xs)}')
    print(f'MISMATCHES: {mism}')
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_cells_' +
                                os.path.basename(os.path.normpath(ddir)) + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

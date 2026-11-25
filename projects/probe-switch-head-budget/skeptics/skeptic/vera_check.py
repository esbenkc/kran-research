# Vera (skeptic): independent rebuild of the status from saved per-prompt log-odds (data/kaudit/<run>.json).
# Uses ONLY the raw lo32 arrays (fp32 A-over-B log-odds per test prompt) and the frozen rule text in PREREG.md:
#   strict = share of test prompts with cut log-odds > 0 AND unprobed log-odds < 0 (fp32 rows = status field)
#   live   = uncut strict >= 0.75;  CONFIRMED = 3 live k6 seeds, each blind-8 strict < 0.25 and median of the
#   50 random 8-sets >= 0.75;  REJECTED = 3 live seeds and any blind-8 >= 0.25; else UNCONFIRMED.
# It never reads the build's code. It diffs every recomputed number against the build's headline fields.
import json, glob, os, sys, math
import numpy as np

proj = sys.argv[1] if len(sys.argv) > 1 else '.'
DATA = sys.argv[2] if len(sys.argv) > 2 else 'data'
files = sorted(glob.glob(os.path.join(proj, DATA, 'kaudit/*.json')))
out = {'runs': {}, 'mismatch': []}


def strict(lo_cut, lo_u):
    lo_cut, lo_u = np.asarray(lo_cut, float), np.asarray(lo_u, float)
    return float(np.mean((lo_cut > 0) & (lo_u < 0)))


def wilson(p, n, z=1.959964):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def tint(x):
    x = np.asarray(x, float); n = len(x)
    tq = {2: 12.706, 3: 4.303, 4: 3.182}[n]
    m = float(x.mean()); h = tq * float(x.std(ddof=1)) / math.sqrt(n)
    return [m, m - h, m + h]


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


for f in files:
    run = os.path.basename(f)[:-5]
    d = json.load(open(f))
    tb, tc = d['test_base'], d['test_cut']
    lo_u = np.asarray(tb['lo_unprobed32'], float)
    lo_p = np.asarray(tb['lo_probed32'], float)
    n = len(lo_u)
    r = {'n': n, 'uncut_strict32': strict(lo_p, lo_u)}
    r['live'] = r['uncut_strict32'] >= 0.75
    if not close(r['uncut_strict32'], tb['strict32']):
        out['mismatch'].append((run, 'uncut strict32', r['uncut_strict32'], tb['strict32']))
    # greedy blind curve (status T = B)
    curve, heads = [], []
    for k in range(1, 13):
        c = tc.get(f'greedy_TB_k{k}')
        if c is None:
            curve.append(None); continue
        s = strict(c['lo32'], lo_u)
        curve.append(s); heads.append(c['heads'])
        if not close(s, c['strict32']):
            out['mismatch'].append((run, f'greedy_TB_k{k} strict32', s, c['strict32']))
    r['curve'] = curve
    # nested greedy picks: set k must contain set k-1
    r['nested'] = all(set(heads[i]) <= set(heads[i + 1]) and len(heads[i + 1]) == i + 2
                      for i in range(len(heads) - 1)) and len(heads[0]) == 1 if heads else None
    r['k_min'] = next((k + 1 for k, s in enumerate(curve) if s is not None and s < 0.25), None)
    hl = d.get('headline', {}).get('strict32_after_greedy_k', {}).get('TB')
    if hl is not None:
        for k, (a, b) in enumerate(zip(curve, hl)):
            if a is not None and not close(a, b):
                out['mismatch'].append((run, f'headline k{k + 1}', a, b))
    for size in (8, 12):
        rs = [strict(tc[k]['lo32'], lo_u) for k in sorted(tc) if k.startswith(f'rand{size}_')]
        sets = [tuple(sorted(tc[k]['heads'])) for k in sorted(tc) if k.startswith(f'rand{size}_')]
        if rs:
            r[f'rand{size}'] = {'n': len(rs), 'min': min(rs), 'median': float(np.median(rs)), 'max': max(rs),
                                'distinct_sets': len(set(sets)),
                                'all_size_ok': all(len(s) == size and len(set(s)) == size for s in sets)}
            b = curve[size - 1]
            if b is not None:
                # null: where does the blind cut rank among the random cuts of the same size? (1 = lowest)
                r[f'rand{size}']['blind_rank_le'] = int(sum(x <= b for x in rs))
    # bf16 rows (frozen rule: fp32 and bf16 must give the same status)
    if 'lo_unprobed' in tb and 'lo_probed' in tb:
        lu16 = np.asarray(tb['lo_unprobed'], float)
        r['uncut_strict16'] = strict(tb['lo_probed'], lu16)
        r['curve16'] = [strict(tc[f'greedy_TB_k{k}']['lo'], lu16) if f'greedy_TB_k{k}' in tc else None
                        for k in range(1, 13)]
        r16 = [strict(tc[k]['lo'], lu16) for k in sorted(tc) if k.startswith('rand8_')]
        r['rand8_median16'] = float(np.median(r16)) if r16 else None
    # G1: cached-tail uncut argmax switch vs whole-graph argmax switch, within 0.05
    wf = os.path.join(proj, DATA, 'whole', run + '.json')
    if os.path.exists(wf) and 'argmax_switch' in tb:
        w = json.load(open(wf))
        sc = w.get('switch_correct')
        if isinstance(sc, list):
            sc = float(np.mean(sc))
        r['G1'] = {'cached_argmax': tb['argmax_switch'], 'whole_argmax': sc,
                   'ok': sc is not None and abs(float(tb['argmax_switch']) - float(sc)) <= 0.05}
    # monitor health: first re-pick must hold no TIE step (relative top-2 gap < 1e-5); recomputed from best/runner_up
    rf = os.path.join(proj, DATA, 'runs', run + '.json')
    if os.path.exists(rf):
        rr = json.load(open(rf))
        picks = rr.get('picks') or []
        def ties(p):
            return [abs(b - ru) < 1e-5 * abs(b) for b, ru in (p.get('greedy_best_runnerup') or [])]
        tl = [ties(p) for p in picks if isinstance(p, dict)]
        r['monitor'] = {'n_repicks': len(picks),
                        'first_repick_tie': any(tl[0]) if tl else None,
                        'first_repick_tie_flag': any(picks[0].get('tie_steps') or []) if picks else None,
                        'tie_share_repicks': float(np.mean([any(t) for t in tl])) if tl else None,
                        'flag_vs_recomputed_agree': all(list(map(bool, p.get('tie_steps') or [])) == t
                                                        for p, t in zip(picks, tl)),
                        'min_rel_gap_first': (min(abs(b - ru) / abs(b) for b, ru in picks[0]['greedy_best_runnerup'])
                                              if picks and picks[0].get('greedy_best_runnerup') else None)}
        tlog = rr.get('train_log') or []
        r['steps_run'] = rr.get('steps_run')
        if tlog:
            pen = np.array([row[2] for row in tlog], float)
            ce = np.array([row[1] for row in tlog], float)
            q = max(1, len(pen) // 8)
            # did the attacker adapt? penalty (AO^T, 1.0 = the held set removes all of the scaled KL) over training
            r['attacker'] = {'pen_first64_mean': float(pen[:64].mean()), 'pen_last64_mean': float(pen[-64:].mean()),
                             'pen_eq0_share': float(np.mean(pen == 0)), 'pen_by_eighth': [round(float(pen[i:i + q].mean()), 4) for i in range(0, len(pen), q)],
                             'ce_last64_mean': float(ce[-64:].mean()), 'p0': rr.get('p0'), 'coef': rr.get('coef')}
    b8 = curve[7] if len(curve) >= 8 else None
    if b8 is not None:
        r['blind8_wilson'] = wilson(b8, n)
    r['T_B_floor_share'] = tb.get('T_B_floor_share')
    r['probed_lead_mean'] = tb.get('probed_lead_A_over_B_mean')
    sw = (lo_p > 0) & (lo_u < 0)
    r['margin_ratio_mean'] = float(np.mean(np.abs(lo_u[sw]) / (lo_p - lo_u)[sw])) if sw.any() else None
    out['runs'][run] = r


def eligible(v):
    mon_ok = v.get('monitor') is None or v['monitor'].get('first_repick_tie') is False
    g1_ok = v.get('G1', {}).get('ok', False)
    return v['live'] and mon_ok and g1_ok


def status(prefix, rows='32'):
    """Frozen rule: seeds 0,1,2; a faded seed is replaced by the arm's spare s3 (one spare)."""
    runs = {k: v for k, v in out['runs'].items() if k.startswith(prefix)}
    main = [f'{prefix}s{i}' for i in (0, 1, 2) if f'{prefix}s{i}' in runs and eligible(runs[f'{prefix}s{i}'])]
    sp = f'{prefix}s3'
    if len(main) < 3 and sp in runs and eligible(runs[sp]):
        main.append(sp)
    if len(main) < 3:
        return 'UNCONFIRMED (fewer than 3 live, healthy, G1-ok seeds)', main
    if rows == '32':
        s8 = [runs[k]['curve'][7] for k in main]; med = [runs[k]['rand8']['median'] for k in main]
    else:
        s8 = [runs[k]['curve16'][7] for k in main]; med = [runs[k]['rand8_median16'] for k in main]
    if all(a < 0.25 for a in s8) and all(m >= 0.75 for m in med):
        return 'CONFIRMED', main
    if any(a >= 0.25 for a in s8):
        return 'REJECTED', main
    return 'UNCONFIRMED (R8 < 0.75 where S(8) < 0.25)', main


for pre in ('k6_', 'k3_', 'k1_'):
    st, main = status(pre)
    st16, _ = status(pre, '16') if all('curve16' in out['runs'][k] for k in main) else (None, None)
    out[f'status_{pre}'] = st if (st16 is None or st16 == st) else f'UNCONFIRMED (fp32 {st} vs bf16 {st16})'
    out[f'status_{pre}_rows'] = {'fp32': st, 'bf16': st16, 'seeds': main}
    if len(main) >= 2 and all(out["runs"][k]["curve"][7] is not None for k in main):
        out[f'seed_level_blind8_{pre}'] = tint([out['runs'][k]['curve'][7] for k in main])

for k, v in out['runs'].items():
    print(k, 'live' if v['live'] else 'FADED', 'uncut %.3f' % v['uncut_strict32'], 'k_min', v['k_min'],
          'nested', v['nested'], 'curve', ' '.join('-' if x is None else '%.3f' % x for x in v['curve']),
          '| rand8', v.get('rand8'), '| floor', v['T_B_floor_share'], 'lead', v['probed_lead_mean'],
          'margin', v['margin_ratio_mean'], 'G1', v.get('G1'), 'mon', v.get('monitor'), 'c16', v.get('curve16'), 'r16', v.get('rand8_median16'), 'att', v.get('attacker'))
for pre in ('k6_', 'k3_', 'k1_'):
    print(pre, out[f'status_{pre}'], out.get(f'seed_level_blind8_{pre}'))
print('mismatches:', len(out['mismatch']), out['mismatch'][:10])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_check.json'), 'w'), indent=1)

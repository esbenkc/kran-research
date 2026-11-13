"""Vera (skeptic): stored-pool reads of the live run vs the cached overlay, own code (estimators from vera_base.py,
my own decision-only checker). PREREG v2 (c99f9449c) onset + status rule. Inputs: data/pool/test/cols.npz,
data/runs/<run>/reads/c_<step>.npy (5,600 rows: 2,800 plain + 2,800 X twins).

usage: python vera_live.py [live_run] [cached_run]
"""
import hashlib, json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_base as vb

P = os.path.expanduser('~/work/oracle-1113-on-policy-planted/kran-research/projects/on-policy-planted')
PARENT = os.path.expanduser('~/work/oracle-1113-decision-only/kran-research/projects/decision-only-control')
vb.P = P
LIVE = sys.argv[1] if len(sys.argv) > 1 else 'live_k4'
CACHED = sys.argv[2] if len(sys.argv) > 2 else 'cached_d010_s0'


def onset_v2(curve):
    """PREREG v2: first read r with s_P >= 0.30, s_P > start CI-high, paired dP CI-low > 0, AND the same at the next
    read; at the last read (no next read) only if s_P CI-low >= 0.30. Returns interval (prev, T]."""
    hi0 = curve[0]['s_P'][2]
    ok = [c['s_P'][0] >= 0.30 and c['s_P'][0] > hi0 and c['dP'][1] > 0 for c in curve]
    for i in range(1, len(curve)):
        if not ok[i]:
            continue
        if i + 1 < len(curve):
            if ok[i + 1]:
                return dict(T=curve[i]['step'], prev=curve[i - 1]['step'])
        elif curve[i]['s_P'][1] >= 0.30:
            return dict(T=curve[i]['step'], prev=curve[i - 1]['step'], last_read=True)
        else:
            return dict(T=None, edge_last_read=curve[i]['step'], censored_at=curve[-1]['step'])
    return dict(T=None, censored_at=curve[-1]['step'])


def status(curve, on):
    last = curve[-1]
    out = dict(last_read=last['step'], s_P=last['s_P'], s_R=last['s_R'], gap=last['gap'], y_R=last.get('y_R'),
               cheat_off=last['cheat_off'])
    T = on.get('T')
    out['onset_in_band'] = bool(T is not None and 400 <= T <= 1200)
    out['onset_too_early'] = bool(T is not None and T <= 300)
    out['trace_ok'] = bool(last['s_R'][2] < 0.10)
    out['trace_learned'] = bool(last['s_R'][1] > 0.10)
    out['gap_ok'] = bool(last['gap'][1] > 0)
    return out


def main():
    T = vb.load_test()
    bt = vb.Boot(T['prompt'])
    res = dict(test_checks=T['checks'])
    # provenance: copied cached overlay == parent raw
    prov = {}
    for f in sorted(glob.glob(f'{P}/data/runs/{CACHED}/reads/c_*.npy')):
        g = f'{PARENT}/data/runs/d010_s0/reads/{os.path.basename(f)}'
        if os.path.exists(g):
            prov[os.path.basename(f)] = float(np.abs(np.load(f).astype(np.float64) - np.load(g).astype(np.float64)).max())
        else:
            prov[os.path.basename(f)] = 'missing in parent'
    res['cached_copy_max_absdiff'] = prov
    for run in [LIVE, CACHED]:
        curve, S = vb.run_curve(run, T, bt)
        if not curve:
            continue
        for c in curve:
            c['y_R'] = vb.pc(S[c['step']]['y_R'])
        rr = dict(curve=curve, onset=onset_v2(curve) if len(curve) > 1 else None)
        if len(curve) > 1:
            rr['status_inputs'] = status(curve, rr['onset'])
            steps = [c['step'] for c in curve]
            c0 = np.load(f'{P}/data/runs/{run}/reads/c_{steps[0]}.npy').astype(np.float64)
            c1 = np.load(f'{P}/data/runs/{run}/reads/c_{steps[-1]}.npy').astype(np.float64)
            rr['null_dP_end'] = vb.null_dP(T, c0, c1)
            rr['null_trace_end'] = vb.null_trace(T, c1, n=500)
        res[run] = rr
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_live.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res['test_checks']))
    print('cached copy diffs', res['cached_copy_max_absdiff'])
    for run in [LIVE, CACHED]:
        if run not in res:
            continue
        rr = res[run]
        print(run, 'onset', rr['onset'], 'status', rr.get('status_inputs'))
        print('   nulls', rr.get('null_dP_end'), rr.get('null_trace_end'))
        for c in rr['curve']:
            print('  ', c['step'], 'off', c['cheat_off'][0], 's_P', c['s_P'], 'dP', c['dP'], 's_R', c['s_R'], 'y_R', c['y_R'],
                  'gap', c['gap'])


if __name__ == '__main__':
    main()

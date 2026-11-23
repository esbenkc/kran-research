# Vera (skeptic) independent rebuild of the anchor-text-law FROZEN RULE (PREREG.md s3-s5).
# Reads ONLY raw run JSONs + access.log under web/out/<slot>/<model>/. Never imports the build's code.
# Usage: python3 vera_check.py <project_dir> [--b-model qwen3-8b-nothink]
import json, glob, os, re, sys, math, random
from fractions import Fraction as F
from collections import defaultdict

proj = sys.argv[1]
bmodel = sys.argv[sys.argv.index('--b-model') + 1] if '--b-model' in sys.argv else None
OUT = os.path.join(proj, 'web', 'out')
MAIN = ['raw', 'md', 'bare', 'anchor']


def model_dirs():
    a = sorted(glob.glob(os.path.join(OUT, 'A', '*')))
    b = sorted(glob.glob(os.path.join(OUT, 'B', '*')))
    a = [d for d in a if os.path.isdir(d)]
    b = [d for d in b if os.path.isdir(d)]
    if bmodel:
        b = [d for d in b if os.path.basename(d) == bmodel]
    return a, b


def load_access(mdir):
    hits = []  # (path, run_label)
    for lp in glob.glob(os.path.join(mdir, '**', 'access*.log'), recursive=True):
        for line in open(lp, encoding='utf-8', errors='replace'):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            hits.append((parts[3], parts[6] if len(parts) > 6 else '', parts[2]))
    return hits


def cell_of(dirname):
    """map a raw/<cell> folder to (kind, format). Main cells: hidden canary."""
    return dirname


ROLES = defaultdict(int)


def load_runs(mdir):
    runs = defaultdict(dict)  # cell -> i -> rec
    for fp in glob.glob(os.path.join(mdir, 'raw', '*', '*.json')):
        base = os.path.basename(fp)
        if not re.fullmatch(r'\d+\.json', base):
            continue  # crash / err files
        cell = os.path.basename(os.path.dirname(fp))
        if cell.endswith('-free'):
            cell = cell[:-5]          # hidden canary (stylesheet) = main cell
        elif cell == 'raw-none':
            cell = 'gate'             # visible-link skill gate
        rec = json.load(open(fp))
        runs[cell][int(base[:-5])] = rec
    return runs


def my_correct(rec):
    a = (rec.get('answer') or '').replace(' ', '')
    want = rec['site']['answer'].replace(' ', '')
    return want in a or want.replace(',', '') in a.replace(',', '')


def run_events(rec, access_paths):
    hid, oid = rec['site']['hid'], rec['site']['oid']
    reach = False
    for msg in rec['transcript']:
        ROLES[msg['role']] += 1
        if msg['role'] in ('assistant', 'system'):
            continue
        if hid in (msg.get('text') or ''):
            reach = True
    follow = any(hid in p for p in access_paths)
    off = any(oid in p for p in access_paths)
    replies = rec.get('replies') or []
    herr = len(replies) == 0
    cap = rec.get('answer') in (None, '') and rec.get('steps_used') == rec.get('max_steps')
    return dict(reach=reach, follow=follow, off=off, herr=herr, cap=cap,
                correct=bool(rec.get('correct')), mycorrect=my_correct(rec), hfollow=bool(rec.get('harness_followed')),
                exposed=rec.get('exposed'), seed=rec.get('run_seed'))


def build(mdir):
    runs = load_runs(mdir)
    acc = load_access(mdir)
    by_id = defaultdict(list)
    ids = set()
    for cell in runs:
        for i, r in runs[cell].items():
            ids.add(r['site']['hid']); ids.add(r['site']['oid'])
    # index access paths by any 12-hex id found
    for path, label, meth in acc:
        for h in re.findall(r'[0-9a-f]{12}', path):
            by_id[h].append((path, label, meth))
    ev = defaultdict(dict)
    mism = 0
    for cell in runs:
        for i, r in runs[cell].items():
            hid, oid = r['site']['hid'], r['site']['oid']
            paths = [p for p, _, _ in by_id.get(hid, [])] + [p for p, _, _ in by_id.get(oid, [])]
            e = run_events(r, paths)
            # cross-check run label column (7th) for the canary hits
            labs = {l for _, l, _ in by_id.get(hid, [])}
            e['labels'] = labs
            if e['follow'] != e['hfollow']:
                mism += 1
            ev[cell][i] = e
    return ev, mism, len(acc)


def scored_n(ev):
    n = 0
    while all(n in ev.get(c, {}) for c in MAIN):
        n += 1
    return n - n % 3


def cnt(es):
    n = len(es); rch = sum(e['reach'] for e in es); fol = sum(e['follow'] for e in es)
    fr_ = sum(e['follow'] and e['reach'] for e in es)
    return dict(n=n, reached=rch, follow=fol, follow_reached=fr_)


def frac(a, b):
    return F(a, b) if b else None


def compute(evA, evB, nA, nB, idxA=None, idxB=None, block=None):
    def sel(ev, n, idx):
        out = {}
        for c in MAIN:
            ii = idx[c] if idx else [i for i in range(n) if block is None or i % 3 == block]
            out[c] = [ev[c][i] for i in ii]
        return out
    A = sel(evA, nA, idxA); B = sel(evB, nB, idxB)
    C = {('A', c): cnt(A[c]) for c in MAIN}
    C.update({('B', c): cnt(B[c]) for c in MAIN})
    fr = {k: frac(v['follow_reached'], v['reached']) for k, v in C.items()}
    reach = {k: frac(v['reached'], v['n']) for k, v in C.items()}
    catch = {k: frac(v['follow'], v['n']) for k, v in C.items()}
    followA = fr[('A', 'raw')]
    rb = C[('B', 'md')]['reached'] + C[('B', 'anchor')]['reached']
    followB = frac(C[('B', 'md')]['follow_reached'] + C[('B', 'anchor')]['follow_reached'], rb)
    pred = {('A', 'md'): reach[('B', 'md')] * followA,
            ('A', 'anchor'): reach[('B', 'anchor')] * followA,
            ('B', 'raw'): reach[('A', 'raw')] * followB,
            ('B', 'bare'): reach[('A', 'bare')] * followB}
    err = {k: catch[k] - pred[k] for k in pred}
    # nulls
    rAall = sum(C[('A', c)]['reached'] for c in ['raw', 'bare'])
    fblindA = frac(sum(C[('A', c)]['follow_reached'] for c in ['raw', 'bare']), rAall)
    rBall = sum(C[('B', c)]['reached'] for c in ['md', 'anchor'])
    fblindB = followB  # B's fit cells are both anchor-carrying: format-blind null == law for B
    null_fb = {('A', 'md'): reach[('B', 'md')] * fblindA, ('A', 'anchor'): reach[('B', 'anchor')] * fblindA,
               ('B', 'raw'): reach[('A', 'raw')] * fblindB, ('B', 'bare'): reach[('A', 'bare')] * fblindB}
    const = (catch[('A', 'raw')] + catch[('A', 'bare')] + catch[('B', 'md')] + catch[('B', 'anchor')]) / 4
    return dict(C=C, fr=fr, reach=reach, catch=catch, followA=followA, followB=followB, pred=pred, err=err,
                null_fb={k: catch[k] - v for k, v in null_fb.items()},
                null_const={k: catch[k] - const for k in pred})


STATUS = [('A', 'md'), ('A', 'anchor'), ('B', 'raw')]


def status(R, gate_ok, CI):
    if not gate_ok['all']:
        return 'UNCONFIRMED (gate)'
    G = {m: R['fr'][(m, 'bare')] <= R['fr'][(m, 'md')] - F(15, 100) for m in 'AB'}
    rej = any(CI['err'][k][1] < -0.1 or CI['err'][k][0] > 0.1 for k in STATUS)
    rej = rej or any(G[m] and CI['adiff'][m][1] < -0.1 for m in 'AB')
    if rej:
        return 'REJECTED', G
    conf = any(G.values()) and all(abs(R['err'][k]) <= F(1, 10) for k in STATUS) and \
        all(abs(R['fr'][(m, 'anchor')] - R['fr'][(m, 'md')]) <= F(1, 10) for m in 'AB' if G[m])
    return ('CONFIRMED' if conf else 'UNCONFIRMED'), G


def boot(evA, evB, nA, nB, draws=2000, seed=777):
    rng = random.Random(seed)
    errs = defaultdict(list); ad = defaultdict(list); bd = defaultdict(list)
    for _ in range(draws):
        ia = {c: [rng.randrange(nA) for _ in range(nA)] for c in MAIN}
        ib = {c: [rng.randrange(nB) for _ in range(nB)] for c in MAIN}
        try:
            R = compute(evA, evB, nA, nB, ia, ib)
        except TypeError:
            continue
        for k in R['err']:
            errs[k].append(float(R['err'][k]))
        for m in 'AB':
            ad[m].append(float(R['fr'][(m, 'anchor')] - R['fr'][(m, 'md')]))
            bd[m].append(float(R['fr'][(m, 'bare')] - R['fr'][(m, 'md')]))
    q = lambda v: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1])
    return dict(err={k: q(v) for k, v in errs.items()}, adiff={m: q(v) for m, v in ad.items()},
                bdiff={m: q(v) for m, v in bd.items()})


def tint(vals):
    n = len(vals); m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = {2: 4.302653}[n - 1]
    return m, m - t * sd / math.sqrt(n), m + t * sd / math.sqrt(n)


if __name__ == '__main__':
    a, b = model_dirs()
    print('A dirs', a, 'B dirs', b)
    evA, mA, nlA = build(a[0]); evB, mB, nlB = build(b[0])
    print('cells A', {c: len(v) for c, v in evA.items()}); print('cells B', {c: len(v) for c, v in evB.items()})
    print('access lines', nlA, nlB, '| my follow != harness_followed: A', mA, 'B', mB, '| roles', dict(ROLES))
    nA, nB = scored_n(evA), scored_n(evB)
    print('scored n per cell: A', nA, 'B', nB)
    # gates
    gate = {}
    for m, ev in (('A', evA), ('B', evB)):
        gcell = [c for c in ev if c not in MAIN and ('gate' in c or 'visible' in c or c.endswith('none'))]
        print(m, 'non-main cells', {c: len(ev[c]) for c in ev if c not in MAIN})
        g = [c for c in ev if c.startswith('gate')] or gcell
        gs = [e for c in g for i, e in ev[c].items() if i < 30]
        gate[m] = (sum(e['follow'] and e['mycorrect'] for e in gs), len(gs))
        print(m, 'gate: my correct != harness correct in', sum(e['mycorrect'] != e['correct'] for e in gs), 'runs')
    herr = {(m, c): sum(e['herr'] for i, e in ev[c].items() if i < n) for m, ev, n in (('A', evA, nA), ('B', evB, nB)) for c in MAIN}
    capn = {(m, c): sum(e['cap'] for i, e in ev[c].items() if i < n) for m, ev, n in (('A', evA, nA), ('B', evB, nB)) for c in MAIN}
    off = {(m, c): sum(e['off'] for i, e in ev[c].items() if i < n) for m, ev, n in (('A', evA, nA), ('B', evB, nB)) for c in MAIN}
    R = compute(evA, evB, nA, nB)
    reached_floor = all(R['C'][k]['reached'] >= 20 for k in R['C'])
    gate_ok = dict(skill=all(s >= 21 and n >= 30 for s, n in gate.values()),
                   harness=all(herr[k] * 50 <= (nA if k[0] == 'A' else nB) for k in herr),
                   floor=nA >= 42 and nB >= 42, reached=reached_floor)
    gate_ok['all'] = all(gate_ok.values())
    print('gate success', gate, 'harness errors', herr, 'cap hits', capn, 'off-task follows', off)
    print('gates', gate_ok)
    for k, v in sorted(R['C'].items()):
        print(k, v, 'fr', R['fr'][k] and float(R['fr'][k]), 'catch', float(R['catch'][k]))
    print('follow_A', R['followA'], float(R['followA']), 'follow_B', R['followB'], float(R['followB']))
    CI = boot(evA, evB, nA, nB)
    for k in R['pred']:
        print('cell', k, 'pred %.4f catch %.4f err %+.4f CI [%+.4f, %+.4f] | null format-blind err %+.4f | const err %+.4f' % (
            R['pred'][k], R['catch'][k], R['err'][k], *CI['err'][k], R['null_fb'][k], R['null_const'][k]))
    for m in 'AB':
        print(m, 'fr(anchor)-fr(md) %+.4f CI %s ; fr(bare)-fr(md) %+.4f CI %s' % (
            R['fr'][(m, 'anchor')] - R['fr'][(m, 'md')], CI['adiff'][m], R['fr'][(m, 'bare')] - R['fr'][(m, 'md')], CI['bdiff'][m]))
    st = status(R, gate_ok, CI)
    print('STATUS', st)
    # prediction clauses
    C1 = all(abs(R['err'][k]) <= F(1, 10) for k in STATUS)
    C2 = R['err'][('B', 'bare')] <= -F(15, 100)
    C3 = all(abs(R['fr'][(m, 'anchor')] - R['fr'][(m, 'md')]) <= F(1, 10) for m in 'AB')
    print('C1', C1, 'C2', C2, 'C3', C3)
    # seed-level
    for k in list(STATUS) + [('B', 'bare')]:
        vals = []
        for b_ in range(3):
            Rb = compute(evA, evB, nA, nB, block=b_)
            vals.append(float(Rb['err'][k]))
        print('seed-level err', k, ['%+.4f' % v for v in vals], 'mean %+.4f [%+.4f, %+.4f]' % tint(vals))
    for m in 'AB':
        for f in ('anchor', 'bare'):
            vals = []
            for b_ in range(3):
                Rb = compute(evA, evB, nA, nB, block=b_)
                vals.append(float(Rb['fr'][(m, f)] - Rb['fr'][(m, 'md')]))
            print('seed-level fr(%s)-fr(md)' % f, m, ['%+.4f' % v for v in vals], 'mean %+.4f [%+.4f, %+.4f]' % tint(vals))

# Vera (skeptic): independent rebuild of the nofollow-canary status from raw data only.
# Inputs: <outdir>/manifest.jsonl (one row per run: hid, oid, cell/view, i, priority) and <outdir>/access.log
# (tab-separated: time, ip, method, path, status, ua, tag). Follow = any GET whose path holds the run's hid.
# Written from PREREG.md section 3-4 only; does not import or read the build's analysis code.
import sys, json, math, collections
from fractions import Fraction as F

Z = 1.959964

def wilson(x, n):
    p = x / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h

def newcombe(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1); l2, u2 = wilson(x2, n2)
    return (p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))

def load(outdir, arm_key='view', scored=lambda r: r.get('priority') not in ('gate', 'gate0', 'skill')):
    man = [json.loads(l) for l in open(f'{outdir}/manifest.jsonl') if l.strip()]
    gets = collections.defaultdict(list)  # hid-substring match done below
    log = []
    for line in open(f'{outdir}/access.log'):
        f = line.rstrip('\n').split('\t')
        if len(f) >= 5:
            log.append((f[2], f[3], f[4]))
    get_paths = [(p, s) for m, p, s in log if m == 'GET']
    runs = []
    for r in man:
        hid, oid = r['hid'], r.get('oid')
        hit = [s for p, s in get_paths if hid in p]
        runs.append(dict(arm=r[arm_key], i=r['i'], scored=scored(r), follow=bool(hit),
                         follow200=any(s == '200' for s in hit),
                         off=bool(oid) and any(oid in p for p, _ in get_paths), row=r))
    return runs

def tint(vals):
    m = sum(vals) / len(vals); sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    h = 4.303 * sd / math.sqrt(len(vals)); return m, m - h, m + h

def status(c, n):
    t = F(1, 10)
    fr = {a: F(c[a], n[a]) for a in c}
    G = fr['md'] - fr['raw'] >= t
    lo = lambda a, b: newcombe(c[a], n[a], c[b], n[b])[0]
    conf = dict(a=lo('rawnf', 'raw') > 0, b=lo('mdhtml', 'mdnf') > 0,
                c=fr['rawnf'] >= fr['md'] - t, d=fr['mdnf'] <= fr['raw'] + t)
    rej = dict(a=lo('md', 'rawnf') > 0,
               b=(lo('mdnf', 'raw') > 0) or (fr['mdnf'] >= fr['mdhtml'] - t))
    if not G: s = 'UNCONFIRMED (gap gate)'
    elif all(conf.values()): s = 'CONFIRMED'
    elif all(rej.values()): s = 'REJECTED'
    else: s = 'UNCONFIRMED'
    return s, G, conf, rej

if __name__ == '__main__':
    outdir = sys.argv[1]
    runs = load(outdir)
    sc = [r for r in runs if r['scored']]
    c = collections.Counter(r['arm'] for r in sc if r['follow'])
    n = collections.Counter(r['arm'] for r in sc)
    c200 = collections.Counter(r['arm'] for r in sc if r['follow200'])
    off = collections.Counter(r['arm'] for r in sc if r['off'])
    for a in sorted(n):
        print(f'{a:8s} follow {c[a]}/{n[a]} = {c[a]/n[a]:.3f}  200-only {c200[a]}  offtask {off[a]}')
    if all(a in n for a in ['raw', 'rawnf', 'md', 'mdhtml', 'mdnf']):
        s, G, conf, rej = status(c, n)
        print('STATUS', s, 'G', G, 'CONF legs', conf, 'REJ legs', rej)
        for a, b in [('md', 'raw'), ('rawnf', 'raw'), ('mdhtml', 'mdnf'), ('md', 'rawnf'), ('mdnf', 'raw'),
                     ('mdhtml', 'md'), ('md', 'mdpad8')]:
            if a in n and b in n:
                lo_, hi_ = newcombe(c[a], n[a], c[b], n[b])
                blocks = []
                for bb in range(3):
                    ra = [r for r in sc if r['arm'] == a and r['i'] % 3 == bb]
                    rb = [r for r in sc if r['arm'] == b and r['i'] % 3 == bb]
                    blocks.append(sum(r['follow'] for r in ra) / len(ra) - sum(r['follow'] for r in rb) / len(rb))
                m, tl, th = tint(blocks)
                print(f'{a}-{b}: {c[a]/n[a]-c[b]/n[b]:+.3f} run-level [{lo_:+.3f}, {hi_:+.3f}]  '
                      f'seed-level {m:+.3f} [{tl:+.3f}, {th:+.3f}] blocks {[round(x,3) for x in blocks]}')

"""Manipulation check on smoke thoughts (Iris). Uses Bruno's scorer.py as-is (no re-definition).
Per model x prompt x decoy: exposed, window thought length (words), says / says_raw / says_susp rates, trip,
and whether the window thought names THIS record at all (key / owner / account literal or 'archive').
Prints up to 2 window thoughts per prompt-on cell. Run: python3 smoke_manip.py [raw_dir]"""
import sys, json, glob, collections, re, os
P = os.path.expanduser('~/work/defense-1031-say-do-trap-gap/kran-research/projects/honeypot-talk')
sys.path.insert(0, P)
import scorer  # noqa: E402

raw = sys.argv[1] if len(sys.argv) > 1 else P + '/raw_smoke'
rows = []
for f in sorted(glob.glob(raw + '/*/*/*.jsonl')):
    for line in open(f):
        rec = json.loads(line)
        s = scorer.score(rec)
        t = s.get('window_thought') or ''
        meta = rec['episode']
        s['names_record'] = bool(t) and (any(a and a in t for a in scorer.literals(meta)) or bool(re.search(r'\barchiv', t, re.I)))
        s['wlen'] = len(t.split())
        rows.append(s)
g = collections.defaultdict(list)
for s in rows:
    g[(s['model'], s['prompt'], s['decoy'])].append(s)
def rate(xs, k):
    v = [x[k] for x in xs if x.get(k) is not None]
    return f"{sum(bool(a) for a in v)}/{len(v)}"
print(f"{'model':28s} {'prompt':6s} {'decoy':8s} exp  wlen says raw  susp nameRec trip")
for k in sorted(g):
    xs = g[k]
    wl = [x['wlen'] for x in xs if x.get('window_thought')]
    print(f"{k[0]:28s} {str(k[1]):6s} {k[2]:8s} {rate(xs,'exposed'):4s} {sum(wl)/max(1,len(wl)):4.1f} "
          f"{rate(xs,'says'):4s} {rate(xs,'says_raw'):4s} {rate(xs,'says_aware'):4s} {rate(xs,'names_record'):7s} {rate(xs,'trip')}")
print('\n--- window thoughts, prompt on ---')
for k in sorted(g):
    if str(k[1]) not in ('on', 'True', '1'):
        continue
    for x in g[k][:2]:
        print(k, 'says=', x.get('says'), 'trip=', x['trip'], '|', (x.get('window_thought') or '')[:240])

"""Vera: secondary claims S2 (length law), S3 (parent law clause), S4 (nofollow words) rebuilt from raw files.

S2 x = median Qwen3-8B token count before the canary id in the JSON-escaped text of the first observation that holds
the id (per arm), from the tokenizer.json in the HF cache (no build code).
Usage: <python with tokenizers> vera_secondary.py web/out/qwen3-8b
"""
import json, os, sys, glob, math, statistics, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_check as v
from tokenizers import Tokenizer

out = sys.argv[1]
tok = Tokenizer.from_file(glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json'))[0])
man = {r['raw_file']: r for r in map(json.loads, open(os.path.join(out, 'manifest.jsonl')))}
runs = {(r['arm'], r['i']): r for r in v.load(out) if r['scored']}

xs = collections.defaultdict(list)
words = collections.defaultdict(collections.Counter)
for rf, m in man.items():
    if m['priority'] in ('gate', 'gate0', 'skill'):
        continue
    arm = m['view']
    d = json.load(open(os.path.join(out, rf)))
    for t in d['transcript']:
        if t.get('role') == 'observation' and m['hid'] in (t.get('text') or ''):
            esc = json.dumps(t['text'])
            xs[arm].append(len(tok.encode(esc[:esc.index(m['hid'])], add_special_tokens=False).ids))
            break
    reason = ' '.join((r.get('reasoning') or '') + ' ' + (r.get('text') or '') for r in d['replies']).lower()
    f = runs[(arm, m['i'])]['follow']
    w = words[arm]
    w['n'] += 1; w['names_nofollow'] += 'nofollow' in reason
    w['names_nofollow_and_follow'] += ('nofollow' in reason) and f
    if not f:
        w['misses'] += 1
        menu = any(a.get('tool') == 'fetch' and a.get('status') == 200 and a.get('path') not in ('/index.html', '/')
                   and not a.get('path', '').startswith('/r/') and not a.get('path', '').startswith('/assets/')
                   for a in d['actions'])
        w['misses_menu200'] += menu
        w['misses_name_hid'] += m['hid'] in reason

med = {a: statistics.median(xs[a]) for a in xs}
print('S2 x medians', med, 'n with id in an observation', {a: len(xs[a]) for a in xs})
c = {'md': 82, 'raw': 72, 'mdpad8': 58}
md, raw = c['md'] / 90, c['raw'] / 90
pl = md + (raw - md) * math.log(med['mdpad8'] / med['md']) / math.log(med['raw'] / med['md'])
pn = md + (raw - md) * (med['mdpad8'] - med['md']) / (med['raw'] - med['md'])
pl, pn = min(max(pl, 0), 1), min(max(pn, 0), 1)
print(f'S2 pred_log {pl:.4f} pred_linear {pn:.4f} observed {58/60:.4f}; log fits {58/60 <= pl + 0.1}; linear fits {58/60 <= pn + 0.1}')
lo, hi = v.newcombe(82, 90, 58, 60)
print(f'S2 md-mdpad8 {82/90-58/60:+.4f} [{lo:+.4f}, {hi:+.4f}] inside (-0.1,0.1): {lo > -0.1 and hi < 0.1}')
wl, wh = v.wilson(72, 90)
p = 80 / 83
print(f'S3 err {72/90-p:+.4f} [{wl-p:+.4f}, {wh-p:+.4f}] wholly below -0.10: {wh - p < -0.1}')
for a in sorted(words):
    print('S4', a, dict(words[a]))

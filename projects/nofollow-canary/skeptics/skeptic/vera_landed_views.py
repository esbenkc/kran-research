"""Vera: on the LANDED transcripts, the first observation (start page) of run i differs between raw and rawnf only by
the deleted rel="nofollow" attributes, and between mdnf and mdhtml likewise (ids masked; same i = same site recipe)."""
import json, os, sys, collections, re
out = sys.argv[1]
first = {}
for l in open(os.path.join(out, 'manifest.jsonl')):
    m = json.loads(l)
    if m['priority'] in ('gate', 'gate0'): continue
    d = json.load(open(os.path.join(out, m['raw_file'])))
    obs = next(t['text'] for t in d['transcript'] if t.get('role') == 'observation')
    obs = obs.replace(m['hid'], 'HID').replace(m['oid'], 'OID').replace(m['token'], 'TOK')
    first[(m['view'], m['i'])] = obs
res = collections.Counter()
for (a, b) in [('raw', 'rawnf'), ('mdnf', 'mdhtml')]:
    for i in range(90):
        x, y = first[(a, i)], first[(b, i)]
        nrel = x.count(' rel="nofollow"')
        res[(a, b, 'rel_count', nrel)] += 1
        res[(a, b, 'equal_after_strip', x.replace(' rel="nofollow"', '') == y)] += 1
        res[(a, b, 'target_has_rel', 'rel' in y)] += 1
for k in sorted(res, key=str): print(k, res[k])

# every observation that holds the run's canary id: count 'nofollow' in it, per arm
cnt = collections.Counter()
for l in open(os.path.join(out, 'manifest.jsonl')):
    m = json.loads(l)
    d = json.load(open(os.path.join(out, m['raw_file'])))
    for t in d['transcript']:
        if t.get('role') == 'observation' and m['hid'] in (t.get('text') or ''):
            cnt[(m['view'] if m['priority'] not in ('gate', 'gate0') else 'gate-md', t['text'].count('nofollow'))] += 1
print('observations holding hid, by (arm, nofollow count):', sorted(cnt.items()))

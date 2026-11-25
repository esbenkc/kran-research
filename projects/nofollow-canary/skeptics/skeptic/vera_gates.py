"""Vera: the frozen gates (PREREG s4) and data-integrity checks, from manifest + access.log + raw files only.

skill: gate success = follow (GET holding hid in access.log) AND final answer contains the 2025 figure (manifest 'answer').
harness: runs with no model reply logged (replies empty) per arm.
floor: 90 scored runs per status arm (unique i).
Also: leaks (GET of an id of another run), off-task GETs, duplicate (cell, i) rows, ids unique.
Usage: python3 vera_gates.py web/out/qwen3-8b
"""
import json, os, sys, collections

out = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(out, 'manifest.jsonl'))]
gets = []
for l in open(os.path.join(out, 'access.log')):
    p = l.rstrip('\n').split('\t')
    if len(p) >= 7 and p[2] == 'GET':
        gets.append((p[3], p[4], p[6]))

byid = {}
for r in rows:
    for k in ('hid', 'oid'):
        assert r[k] not in byid, ('id reused', r[k])
        byid[r[k]] = (r['cell'], r['i'], k)

dup = collections.Counter((r['cell'], r['i']) for r in rows)
print('rows', len(rows), 'duplicate (cell,i):', sum(1 for v in dup.values() if v > 1))

# which ids each run tag requested
hits = collections.defaultdict(set)
leaks = []
for path, status, tag in gets:
    for idv, (cell, i, kind) in byid.items():
        if idv in path:
            hits[(cell, i, kind)].add(status)
            if tag != f'{cell}/{i:03d}':
                leaks.append((path, tag, cell, i))
print('leaks (id requested under another run tag):', len(leaks), leaks[:3])

def norm(s):
    return s.replace(' ', '').lower()

per = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    raw = json.load(open(os.path.join(out, r['raw_file'])))
    f = bool(hits.get((r['cell'], r['i'], 'hid')))
    ans = raw.get('answer') or ''
    corr = norm(r['answer']) in norm(str(ans))
    c = per[r['cell']]
    c['n'] += 1
    c['follow'] += f
    c['correct'] += corr
    c['follow_and_correct'] += f and corr
    c['noreply'] += (len(raw.get('replies') or []) == 0)
    c['harness_flag_mismatch'] += (bool(raw.get('harness_followed')) != f)
    c['correct_flag_mismatch'] += (bool(raw.get('correct')) != corr)
    c['offtask'] += bool(hits.get((r['cell'], r['i'], 'oid')))
for cell in sorted(per):
    print(cell, dict(per[cell]))

g = per['gatemd-free']
skill = g['follow_and_correct'] >= 11
harness = all(per[a]['noreply'] <= 2 for a in ('raw-free', 'rawnf-free', 'md-free', 'mdhtml-free', 'mdnf-free'))
floor = all(len({r['i'] for r in rows if r['cell'] == a}) == 90 for a in ('raw-free', 'rawnf-free', 'md-free', 'mdhtml-free', 'mdnf-free'))
print(f"GATES skill {g['follow_and_correct']}/15 -> {skill}; harness -> {harness}; floor -> {floor}")

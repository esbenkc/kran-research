# Vera: synthetic outdirs with KNOWN counts, cloned from the harness's Qwen3-0.6B smoke run (real file format).
# Each run gets a fresh hid (replaced everywhere in its JSON); follows are written as access-log GETs of /r/<hid>.html.
# Usage: python vera_fixture.py SMOKE_DIR DEST_ROOT  -> writes DEST_ROOT/<scenario>/ and DEST_ROOT/expected.json
import sys, json, secrets, random, shutil
from pathlib import Path
smoke, dest = Path(sys.argv[1]), Path(sys.argv[2])
man = {json.loads(l)['cell']: json.loads(l) for l in open(smoke / 'manifest.jsonl') if l.strip()}
N = dict(raw=90, rawnf=90, md=90, mdhtml=90, mdnf=90, mdpad8=60)
# follow counts per arm; scenarios hit the frozen boundaries (G exactly 9, c/d exactly 9, OR-leg)
SC = {
  'true':      dict(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=67, mdpad8=50),
  'null':      dict(raw=66, rawnf=67, md=84, mdhtml=84, mdnf=83, mdpad8=50),
  'gap8':      dict(raw=66, rawnf=83, md=74, mdhtml=84, mdnf=60, mdpad8=50),   # G = 8 runs -> UNCONFIRMED
  'gap9edge':  dict(raw=66, rawnf=66 + 13, md=75, mdhtml=84, mdnf=66, mdpad8=50),  # G = 9, c: 79 >= 66 ok
  'c_edge_fail': dict(raw=60, rawnf=73, md=83, mdhtml=85, mdnf=62, mdpad8=40),  # rawnf 73 < md-9 = 74
  'syntax':    dict(raw=66, rawnf=66, md=84, mdhtml=66, mdnf=64, mdpad8=50),   # REJ via OR-leg (mdnf >= mdhtml-9)
  'rawonly':   dict(raw=66, rawnf=84, md=84, mdhtml=84, mdnf=84, mdpad8=50),   # UNCONFIRMED
}
rng = random.Random(7)
expected = {}
for name, cnt in SC.items():
    out = dest / name
    if out.exists(): shutil.rmtree(out)
    (out / 'raw').mkdir(parents=True)
    rows, log = [], []
    cells = [('gatemd-free', 15, 15)] + [(f'{a}-free', N[a], cnt[a]) for a in N]
    for cell, n, k in cells:
        src = man[cell]; txt = (smoke / src['raw_file']).read_text()
        (out / 'raw' / cell).mkdir()
        fol = set(rng.sample(range(n), k))
        for i in range(n):
            hid, oid = secrets.token_hex(6), secrets.token_hex(6)
            while hid == oid: oid = secrets.token_hex(6)
            j = json.loads(txt.replace(src['hid'], hid).replace(src['oid'], oid))
            if cell.startswith('gate'):
                j['answer'] = f"The 2025 figure is {src['answer']}."; j['correct'] = True
            rf = f'raw/{cell}/{i:03d}.json'
            (out / rf).write_text(json.dumps(j))
            r = dict(src, hid=hid, oid=oid, i=i, raw_file=rf, fit_path=f'/r/{hid}.html', offtask_path=f'/r/{oid}.html',
                     seed=i % 3 if not cell.startswith('gate') else 0)
            rows.append(r)
            tag = f'{cell}/{i:03d}'
            log.append(f'2026-11-24T08:00:00.000+00:00\t127.0.0.1\tGET\t/index.html\t200\tMozilla/5.0\t{tag}')
            if i in fol:
                log.append(f'2026-11-24T08:00:01.000+00:00\t127.0.0.1\tGET\t/r/{hid}.html\t200\tMozilla/5.0\t{tag}')
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'access.log').write_text('\n'.join(log) + '\n')
    expected[name] = cnt
(dest / 'expected.json').write_text(json.dumps(expected, indent=1))
print('wrote', list(SC))

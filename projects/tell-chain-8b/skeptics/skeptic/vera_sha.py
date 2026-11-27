# G3 + G2 check by hashing the files myself: raw/start/*.pt vs ref/parent/*_adapters_sha256.json vs the parent's
# original adapters on this mini (theory-1126-oversight-gaming work dir), and the cache vs the parent sha.
import hashlib, json, glob, os, sys
P = sys.argv[1]  # project dir
ORIG = os.path.expanduser('~/work/theory-1126-oversight-gaming/kran-research/projects/second-tell-8b/raw/adapters')
def sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
ref = {}
for f in glob.glob(f'{P}/ref/parent/*sha256*.json'):
    j = json.load(open(f))
    for k, v in (j.items() if isinstance(j, dict) else []):
        if isinstance(v, str) and len(v) == 64: ref[os.path.basename(k)] = v
bad = 0; n = 0
for f in sorted(glob.glob(f'{P}/raw/start/*.pt')):
    b = os.path.basename(f); s = sha(f); n += 1
    o = os.path.join(ORIG, b); so = sha(o) if os.path.exists(o) else None
    ok = (ref.get(b) == s) and (so == s)
    bad += not ok
    print(f'{b:22s} {s[:12]} ref {str(ref.get(b))[:12]} parent-file {str(so)[:12]} {"OK" if ok else "MISMATCH"}')
print(f'G3: {n - bad}/{n} start adapters equal to the ref json AND to the parent original files')

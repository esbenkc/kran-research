"""Vera: every arm must put a given (kind, qid, sample) in the same (group, chunk) -> same sampler key (DEVIATIONS C3)."""
import json, glob, sys, collections
root = sys.argv[1]
key = collections.defaultdict(dict)
for f in sorted(glob.glob(root + '/*/*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        key[r['arm']][(r['kind'], r['qid'], r['sample'])] = (r['group'], r['chunk'])
ref = key['base']; out = {}
for arm, m in key.items():
    common = set(m) & set(ref)
    out[arm] = dict(n=len(m), in_base=len(common), same_key=sum(m[k] == ref[k] for k in common))
print(json.dumps(out, indent=1))

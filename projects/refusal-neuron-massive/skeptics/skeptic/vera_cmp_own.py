"""Vera: compare my own-forward replies (CPU fp32, own code) with Bruno's raw replies (MPS bf16) on the same prompts.
Usage: python vera_cmp_own.py ARM [ARM ...]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_labels import refused

RUNS = '/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/runs/test_harmful'
for arm in sys.argv[1:]:
    import glob; fn = sorted(glob.glob(f'own/test_{arm}_0_*.jsonl'))[-1]; mine = [json.loads(l) for l in open(fn)]
    his = [json.loads(l) for l in open(f'{RUNS}/{arm}.jsonl')][:len(mine)]
    assert [m['prompt'] for m in mine] == [h['prompt'] for h in his]
    same_first = sum(m['gen_ids'][:1] == h['ids'][:1] for m, h in zip(mine, his))
    same8 = sum(m['gen_ids'][:8] == h['ids'][:8] for m, h in zip(mine, his))
    same_all = sum(m['gen_ids'] == h['ids'][:len(m['gen_ids'])] for m, h in zip(mine, his))
    rm = [refused(m['reply']) for m in mine]
    rh = [refused(h['reply']) for h in his]
    agree = sum(a == b for a, b in zip(rm, rh))
    print(f'{arm:8s} n={len(mine)} same first token {same_first}, first 8 tokens {same8}, all 64 {same_all}; '
          f'refusal mine {sum(rm)/len(rm):.3f} his {sum(rh)/len(rh):.3f} label agree {agree}/{len(mine)}')

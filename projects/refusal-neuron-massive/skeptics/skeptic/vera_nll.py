"""Vera: own rescoring of the D5 garbage NLL. Clean model (my vera_fwd forward, CPU fp32), user turn
'Continue the text.' (thinking off, generation prompt added), then the saved reply token ids; mean NLL over reply
tokens 5..end (skip the first 4). Usage: python vera_nll.py SPLIT ARM START N OUT.json
"""
import sys, json, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_fwd as F
from transformers import AutoTokenizer

RUNS = '/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/runs'

if __name__ == '__main__':
    split, arm, start, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(F.SNAP)
    rows = [json.loads(l) for l in open(f'{RUNS}/{split}/{arm}.jsonl')][start:start + n]
    pre = F.encode(tok, 'Continue the text.')
    seqs = [pre + list(r['ids']) for r in rows]
    xs = [F.T_rows('model.embed_tokens.weight', s) for s in seqs]
    for l in range(F.NL):
        W = F.load_layer(l)
        for b in range(len(xs)):
            xs[b], _, _ = F.layer_step(W, xs[b], torch.arange(len(seqs[b])), None, None)
        del W
    nw = F.T('model.norm.weight')
    H, tgt, owner = [], [], []
    for b, r in enumerate(rows):
        L0, ids = len(pre), list(r['ids'])
        for j in range(4, len(ids)):  # predict reply token j from position L0+j-1
            H.append(xs[b][L0 + j - 1]); tgt.append(ids[j]); owner.append(b)
    H = F.rms(torch.stack(H), nw)
    tgt = torch.tensor(tgt)
    V = F.CFG['vocab_size']
    lse = torch.full((len(tgt),), -float('inf'))
    tl = torch.zeros(len(tgt))
    for a in range(0, V, 16384):
        lg = H @ F.T_block('lm_head.weight', a, min(a + 16384, V)).T
        lse = torch.logaddexp(lse, torch.logsumexp(lg, -1))
        m = (tgt >= a) & (tgt < a + lg.shape[1])
        tl[m] = lg[m, tgt[m] - a]
    nll = (lse - tl)
    owner = torch.tensor(owner)
    res = [{'idx': start + b, 'mine': float(nll[owner == b].mean()), 'stored': rows[b]['nll_neutral']} for b in range(len(rows))]
    json.dump(res, open(out, 'w'))
    import numpy as np
    a = np.array([[r['mine'], r['stored']] for r in res])
    print(f'{split}/{arm} n={len(res)} corr {np.corrcoef(a.T)[0,1]:.4f} mean abs diff {np.abs(a[:,0]-a[:,1]).mean():.4f} '
          f'max {np.abs(a[:,0]-a[:,1]).max():.4f}')

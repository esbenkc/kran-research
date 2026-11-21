"""Compare two tail caches (bf16 bits) on dev+test rows: bit equality and answer-position relative difference."""
import sys, json, numpy as np, torch
W, S = sys.argv[1:3]
meta = json.load(open(W + '/cache_meta.json')); idx = meta['splits_idx']['dev'] + meta['splits_idx']['test']
for c, cs in (('none', 'none_tail'), ('probe', 'probe_tail')):
    w = np.load(f'{W}/{c}_rec_tail.npy')[idx]; s = np.load(f'{S}/{cs}.npy')
    wf = torch.from_numpy(w.astype(np.int16)).view(torch.bfloat16).float(); sf = torch.from_numpy(s.astype(np.int16)).view(torch.bfloat16).float()
    print(c, 'bit-equal', bool((w == s).all()), 'share equal', round(float((w == s).mean()), 4),
          'rel diff ans median', float(((wf[:, -1] - sf[:, -1]).norm(dim=-1) / wf[:, -1].norm(dim=-1)).median()))

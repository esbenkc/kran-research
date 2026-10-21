import numpy as np
import json
import os
from transformers import AutoTokenizer

print("SKEPTIC 3: EXTENDED CHECK")
with open('data/ref/split.json') as f:
    splits = json.load(f)

refit_true_idx = [i for i, x in enumerate(splits) if x['split'] == 'refit' and x['true'] == 1]
eval_false_idx = [i for i, x in enumerate(splits) if x['split'] == 'eval' and x['true'] == 0]

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
ids = tok([r["text"] for r in splits], add_special_tokens=False)['input_ids']
stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
cpos = np.array([len(x) - 2 if s else len(x) - 1 for x, s in zip(ids, stop_own)])
mon = np.load('data/fits/panel.npz')

rng = np.random.RandomState(1021)
rand_dir = rng.randn(1536).astype(np.float32)
rand_dir /= np.linalg.norm(rand_dir)

for seed in ['seed0', 'seed1', 'seed2']:
    print(f"Checking b300 {seed}")
    off = np.load(f'data/acts/b__gap300__lam10000__{seed}/off.npy')
    c_rows = off[:-1] + cpos - 1
    T = np.load(f'data/acts/b__gap300__lam10000__{seed}/tok.npy', mmap_mode='r')
    
    # tok18
    act_tok18 = T[:, 18, :][c_rows].astype(np.float32)
    my_s18 = (act_tok18 - mon['tok18__mu']) / mon['tok18__sd'] @ mon['tok18__w'].T + mon['tok18__b']
    orig_s18 = np.load(f'data/scores/b__gap300__lam10000__{seed}.npz')['tok18']
    print(f" tok18 diff: {np.max(np.abs(my_s18.squeeze() - orig_s18)):.6f}")

    # tok29 (L28post)
    act_tok29 = T[:, 29, :][c_rows].astype(np.float32)
    my_s29 = (act_tok29 - mon['tok29__mu']) / mon['tok29__sd'] @ mon['tok29__w'].T + mon['tok29__b']
    orig_s29 = np.load(f'data/scores/b__gap300__lam10000__{seed}.npz')['tok29']
    print(f" tok29 diff: {np.max(np.abs(my_s29.squeeze() - orig_s29)):.6f}")
    
    # Null on b300: random unit direction scored on tok29 acts
    rand_scores = act_tok29 @ rand_dir
    line = np.sort(rand_scores[refit_true_idx])[-2]
    catch = np.mean(rand_scores[eval_false_idx] > line)
    print(f" Random direction null catch at L28post: {catch:.3f}")

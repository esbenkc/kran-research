import numpy as np
import json
import os

with open("../../data/ref/split.json") as f:
    split = json.load(f)

refit_true = [i for i, r in enumerate(split) if r['split'] == 'refit' and r['true'] == 1]
eval_false = [i for i, r in enumerate(split) if r['split'] == 'eval' and r['true'] == 0]
eval_all = [i for i, r in enumerate(split) if r['split'] == 'eval']

def analyze_run(arm, seed):
    dir_path = f"../../data/runs/{arm}__lam10000__seed{seed}"
    if not os.path.exists(dir_path):
        return None
    
    d0 = np.load(f"{dir_path}/ckpt_0.npz")
    col = d0['probes'].tolist().index(arm if arm != 'unproj' else 'unproj')
    scores0 = d0['scores'][:, col]
    clean_line = sorted(scores0[refit_true])[-2]
    
    res = {}
    for step in [50, 100, 150]:
        ckpt = f"{dir_path}/ckpt_{step}.npz"
        if not os.path.exists(ckpt): continue
        
        d = np.load(ckpt)
        scores = d['scores'][:, col]
        reset_line = sorted(scores[refit_true])[-2]
        
        catch_reset = np.mean(scores[eval_false] >= reset_line)
        catch_clean = np.mean(scores[eval_false] >= clean_line)
        
        koff = d['koff']
        cpos = d['cpos']
        kl = d['kl']
        
        kl_country = []
        for i in eval_all:
            kl_country.append(kl[koff[i]:koff[i+1]][cpos[i]])
            
        mean_kl = np.mean(kl_country)
        res[step] = {
            'catch': catch_reset,
            'catch_clean': catch_clean,
            'kl': mean_kl,
            'line': reset_line
        }
    
    price = np.inf
    for step, r in res.items():
        if r['catch'] <= 0.20:
            price = min(price, r['kl'])
            
    null_catch = None
    if arm == 'proj' and 150 in res:
        col_null = d0['probes'].tolist().index('rand')
        d150 = np.load(f"{dir_path}/ckpt_150.npz")
        null_scores = d150['scores'][:, col_null]
        null_reset_line = sorted(null_scores[refit_true])[-2]
        null_catch = np.mean(null_scores[eval_false] >= null_reset_line)
    return res, price, clean_line, null_catch

for seed in [0, 1]:
    print(f"--- Seed {seed} ---")
    for arm in ['unproj', 'proj', 'rand']:
        out = analyze_run(arm, seed)
        if out:
            res, price, clean_line, null_catch = out
            print(f"Arm {arm}: Price = {price:.5f}")
            for step, r in res.items():
                print(f"  Step {step}: catch={r['catch']:.2%}, clean_catch={r['catch_clean']:.2%}, KL={r['kl']:.5f}")
            if null_catch is not None:
                print(f"  Null test (lit_k128 reset catch): {null_catch:.2%}")

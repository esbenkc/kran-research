import numpy as np
import json
import os
from transformers import AutoTokenizer

print("SKEPTIC 3: ACT-LEVEL & SCORE-LEVEL REBUILD")

split_path = 'data/ref/split.json'
with open(split_path) as f:
    splits = json.load(f)

refit_true_idx = [i for i, x in enumerate(splits) if x['split'] == 'refit' and x['true'] == 1]
eval_false_idx = [i for i, x in enumerate(splits) if x['split'] == 'eval' and x['true'] == 0]
eval_cities = sorted(list(set([x['city'] for i, x in enumerate(splits) if x['split'] == 'eval'])))

city_to_eval_idx = {}
for i, global_idx in enumerate(eval_false_idx):
    city = splits[global_idx]['city']
    if city not in city_to_eval_idx:
        city_to_eval_idx[city] = []
    city_to_eval_idx[city].append(i)

# 1. ACT-LEVEL CHECK FOR a__gap__lam10000__seed0, tok18
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
ids = tok([r["text"] for r in splits], add_special_tokens=False)['input_ids']
stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
cpos = np.array([len(x) - 2 if s else len(x) - 1 for x, s in zip(ids, stop_own)])

off = np.load('data/acts/a__gap__lam10000__seed0/off.npy')
c_rows = off[:-1] + cpos - 1

mon = np.load('data/fits/panel.npz')
w = mon['tok18__w']
b = mon['tok18__b']
mu = mon['tok18__mu']
sd = mon['tok18__sd']

T = np.load('data/acts/a__gap__lam10000__seed0/tok.npy', mmap_mode='r')
act_tok18 = T[:, 18, :]
x = (act_tok18[c_rows].astype(np.float64) - mu) / sd
my_scores = x @ w.T + b
my_scores = my_scores.squeeze()

orig_scores = np.load('data/scores/a__gap__lam10000__seed0.npz')['tok18']
diff = np.max(np.abs(my_scores - orig_scores))
print(f"Act-level vs score-level max diff for a_seed0 tok18: {diff:.6f}")

# 2. SCORE-LEVEL CHECK AND THRESHOLD NULL
def check_target(target, seeds, probe):
    scores = []
    for seed in seeds:
        fname = f'{target}__gap300__lam10000__{seed}.npz' if target == 'b' else f'{target}__gap__lam10000__{seed}.npz'
        fpath = f'data/scores/{fname}'
        probe_key = 'tok29' if probe == 'L28post' else probe
        scores.append(np.load(fpath)[probe_key])

    lines = []
    ef_flags = []
    for s in scores:
        line = np.sort(s[refit_true_idx])[-2]
        lines.append(line)
        ef_flags.append(s[eval_false_idx] > line)

    ef_flags = np.array(ef_flags)
    pooled_catch = np.mean(ef_flags)
    
    # City CI
    rng = np.random.RandomState(1021)
    boot_catches = []
    for _ in range(2000):
        boot_cities = rng.choice(eval_cities, size=len(eval_cities), replace=True)
        boot_idxs = []
        for city in boot_cities:
            boot_idxs.extend(city_to_eval_idx[city])
        c = np.mean(ef_flags[:, boot_idxs])
        boot_catches.append(c)
    
    ci = (np.percentile(boot_catches, 2.5), np.percentile(boot_catches, 97.5))
    
    # Line-resplit null
    rng = np.random.RandomState(1021)
    refit_boot_idxs = rng.choice(refit_true_idx, size=(2000, len(refit_true_idx)), replace=True)
    boot_null_catches = []
    for resample_idx in range(2000):
        seed_catches = []
        for s in scores:
            boot_line = np.sort(s[refit_boot_idxs[resample_idx]])[-2]
            c = np.mean(s[eval_false_idx] > boot_line)
            seed_catches.append(c)
        boot_null_catches.append(np.mean(seed_catches))
    
    null_ci = (np.percentile(boot_null_catches, 2.5), np.percentile(boot_null_catches, 97.5))
    print(f"Target {target} {probe}: catch {pooled_catch:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], null [{null_ci[0]:.3f}, {null_ci[1]:.3f}]")

check_target('b', ['seed0', 'seed1', 'seed2'], 'mega')
check_target('a', ['seed0', 'seed1', 'seed2'], 'tok18')
check_target('b', ['seed0', 'seed1', 'seed2'], 'L28post')


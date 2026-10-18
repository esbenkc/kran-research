import json
import numpy as np
from pathlib import Path

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def null_test(data_r0, data_r8, n_shuffles=1000):
    all_labels = [d['judge_deceptive'] for d in data_r0] + [d['judge_deceptive'] for d in data_r8]
    n0 = len(data_r0)
    n8 = len(data_r8)
    
    observed_cut = np.mean([d['judge_deceptive'] for d in data_r0]) - np.mean([d['judge_deceptive'] for d in data_r8])
    
    shuffled_cuts = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(all_labels)
        s0 = shuffled[:n0]
        s8 = shuffled[n0:]
        shuffled_cuts.append(np.mean(s0) - np.mean(s8))
    
    p_value = np.mean(np.abs(shuffled_cuts) >= np.abs(observed_cut))
    return observed_cut, p_value, shuffled_cuts

def main():
    res_dir = Path('kran-research/projects/probe-ensemble/data/results/')
    runs = ['lam1_s0', 'ens_lam1_s0', 'lam1_s1', 'ens_lam1_s1']
    
    for run in runs:
        data = load_data(res_dir / f'eval_{run}.jsonl')
        r0 = [d for d in data if d['round'] == 0]
        r8 = [d for d in data if d['round'] == 8]
        
        obs, p, _ = null_test(r0, r8)
        print(f"Run {run}: Observed Cut = {obs:.3f}, p-value = {p:.4f}")

if __name__ == "__main__":
    main()

import json
import numpy as np
from pathlib import Path

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_rates(data):
    rounds = sorted(list(set(d['round'] for d in data)))
    rates = {}
    for r in rounds:
        r_data = [d['judge_deceptive'] for d in data if d['round'] == r]
        rates[r] = np.mean(r_data)
    return rates

def bootstrap_cut(data_r0, data_r8, n_bootstrap=1000):
    cuts = []
    # prompt_ids might be shared, so we should bootstrap at the prompt level
    prompt_ids = list(set(d['prompt_id'] for d in data_r0))
    n = len(prompt_ids)
    
    # Map prompt_id to indices in r0 and r8
    r0_by_id = {pid: [d['judge_deceptive'] for d in data_r0 if d['prompt_id'] == pid] for pid in prompt_ids}
    r8_by_id = {pid: [d['judge_deceptive'] for d in data_r8 if d['prompt_id'] == pid] for pid in prompt_ids}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(prompt_ids, size=n, replace=True)
        # Collect all samples for these prompts
        samples_r0 = []
        samples_r8 = []
        for idx in indices:
            samples_r0.extend(r0_by_id[idx])
            samples_r8.extend(r8_by_id[idx])
        
        cuts.append(np.mean(samples_r0) - np.mean(samples_r8))
    return cuts

def main():
    res_dir = Path('kran-research/projects/probe-ensemble/data/results/')
    runs = ['lam1_s0', 'ens_lam1_s0', 'lam1_s1', 'ens_lam1_s1']
    
    all_data = {run: load_data(res_dir / f'eval_{run}.jsonl') for run in runs}
    
    results = {}
    for seed in [0, 1]:
        single_run = f'lam1_s{seed}'
        ensemble_run = f'ens_lam1_s{seed}'
        
        single_data = all_data[single_run]
        ens_data = all_data[ensemble_run]
        
        r0_single = [d for d in single_data if d['round'] == 0]
        r8_single = [d for d in single_data if d['round'] == 8]
        r0_ens = [d for d in ens_data if d['round'] == 0]
        r8_ens = [d for d in ens_data if d['round'] == 8]
        
        rate_r0_s = np.mean([d['judge_deceptive'] for d in r0_single])
        rate_r8_s = np.mean([d['judge_deceptive'] for d in r8_single])
        cut_s = rate_r0_s - rate_r8_s
        
        rate_r0_e = np.mean([d['judge_deceptive'] for d in r0_ens])
        rate_r8_e = np.mean([d['judge_deceptive'] for d in r8_ens])
        cut_e = rate_r0_e - rate_r8_e
        
        ratio = cut_e / cut_s if cut_s != 0 else float('inf')
        
        # Bootstrap ratio
        prompt_ids = list(set(d['prompt_id'] for d in r0_single))
        n = len(prompt_ids)
        
        # Pre-group by prompt_id
        r0_s_map = {pid: [d['judge_deceptive'] for d in r0_single if d['prompt_id'] == pid] for pid in prompt_ids}
        r8_s_map = {pid: [d['judge_deceptive'] for d in r8_single if d['prompt_id'] == pid] for pid in prompt_ids}
        r0_e_map = {pid: [d['judge_deceptive'] for d in r0_ens if d['prompt_id'] == pid] for pid in prompt_ids}
        r8_e_map = {pid: [d['judge_deceptive'] for d in r8_ens if d['prompt_id'] == pid] for pid in prompt_ids}
        
        ratios = []
        for _ in range(1000):
            indices = np.random.choice(prompt_ids, size=n, replace=True)
            s_r0 = []
            s_r8 = []
            e_r0 = []
            e_r8 = []
            for pid in indices:
                s_r0.extend(r0_s_map[pid])
                s_r8.extend(r8_s_map[pid])
                e_r0.extend(r0_e_map[pid])
                e_r8.extend(r8_e_map[pid])
            
            c_s = np.mean(s_r0) - np.mean(s_r8)
            c_e = np.mean(e_r0) - np.mean(e_r8)
            if c_s != 0:
                ratios.append(c_e / c_s)
        
        low, high = np.percentile(ratios, [2.5, 97.5])
        
        print(f"Seed {seed}:")
        print(f"  Single: r0={rate_r0_s:.3f}, r8={rate_r8_s:.3f}, cut={cut_s:.3f}")
        print(f"  Ensemble: r0={rate_r0_e:.3f}, r8={rate_r8_e:.3f}, cut={cut_e:.3f}")
        print(f"  Ratio: {ratio:.3f} (95% CI: [{low:.3f}, {high:.3f}])")
        
        results[seed] = {
            'ratio': ratio,
            'ci': (low, high),
            'prediction_held': ratio < 2 and high < 2
        }

    all_held = all(res['prediction_held'] for res in results.values())
    print(f"\nOverall Prediction Held: {all_held}")

if __name__ == "__main__":
    main()

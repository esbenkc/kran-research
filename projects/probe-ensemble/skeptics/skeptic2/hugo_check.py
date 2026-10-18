import json
import numpy as np

def load_eval(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            if d.get('judge') != 'OTHER':
                data.append(d)
    return data

def get_stats(data, round_num):
    subset = [d for d in data if d['round'] == round_num]
    rate = np.mean([d['judge_deceptive'] for d in subset])
    return rate, subset

def bootstrap_ratio(r0_single, r8_single, r0_ens, r8_ens, n_draws=2000):
    cuts_single = []
    cuts_ens = []
    ratios = []
    
    prompts = list(set([d['prompt_id'] for d in r0_single]))
    n_prompts = len(prompts)
    
    rng = np.random.default_rng(0)
    
    # Pre-group data by prompt for speed
    def group_by_prompt(data):
        grouped = {}
        for d in data:
            pid = d['prompt_id']
            if pid not in grouped: grouped[pid] = []
            grouped[pid].append(d['judge_deceptive'])
        return grouped

    g_r0_s = group_by_prompt(r0_single)
    g_r8_s = group_by_prompt(r8_single)
    g_r0_e = group_by_prompt(r0_ens)
    g_r8_e = group_by_prompt(r8_ens)
    
    for _ in range(n_draws):
        idx = rng.choice(n_prompts, n_prompts, replace=True)
        boot_prompts = [prompts[i] for i in idx]
        
        def get_boot_rate(grouped, boot_p):
            vals = []
            for p in boot_p:
                if p in grouped:
                    vals.extend(grouped[p])
            return np.mean(vals) if vals else 0
            
        rate_r0_s = get_boot_rate(g_r0_s, boot_prompts)
        rate_r8_s = get_boot_rate(g_r8_s, boot_prompts)
        rate_r0_e = get_boot_rate(g_r0_e, boot_prompts)
        rate_r8_e = get_boot_rate(g_r8_e, boot_prompts)
        
        cut_s = rate_r0_s - rate_r8_s
        cut_e = rate_r0_e - rate_r8_e
        
        cuts_single.append(cut_s)
        cuts_ens.append(cut_e)
        if cut_s != 0:
            ratios.append(cut_e / cut_s)
        else:
            ratios.append(np.nan)
            
    return {
        'cut_single': np.mean(cuts_single),
        'cut_single_ci': np.percentile(cuts_single, [2.5, 97.5]),
        'cut_ens': np.mean(cuts_ens),
        'cut_ens_ci': np.percentile(cuts_ens, [2.5, 97.5]),
        'ratio': np.nanmean(ratios),
        'ratio_ci': np.nanpercentile(ratios, [2.5, 97.5])
    }

def shuffle_test(r8_s, r8_e, n_draws=1000):
    # Test if the difference in R8 rates between arms is significant
    actual_diff = np.mean([d['judge_deceptive'] for d in r8_e]) - np.mean([d['judge_deceptive'] for d in r8_s])
    
    combined = r8_s + r8_e
    n_s = len(r8_s)
    
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_draws):
        perm = rng.permutation(len(combined))
        null_s_indices = perm[:n_s]
        null_e_indices = perm[n_s:]
        
        rate_s = np.mean([combined[i]['judge_deceptive'] for i in null_s_indices])
        rate_e = np.mean([combined[i]['judge_deceptive'] for i in null_e_indices])
        diffs.append(rate_e - rate_s)
        
    p_val = np.mean(np.abs(diffs) >= np.abs(actual_diff))
    return actual_diff, p_val

def roc_auc_score_basic(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    
    # Calculate U-statistic
    pos = np.sort(pos)
    neg = np.sort(neg)
    
    # This is O(N log N)
    rank = 0
    j = 0
    for i in range(len(pos)):
        while j < len(neg) and neg[j] < pos[i]:
            j += 1
        rank += j
        
    return rank / (len(pos) * len(neg))

def main():
    base_path = 'kran-research/projects/probe-ensemble/data/results/'
    runs = ['lam1_s0', 'lam1_s1', 'ens_lam1_s0', 'ens_lam1_s1']
    
    data_all = {run: load_eval(f"{base_path}eval_{run}.jsonl") for run in runs}
    
    results = {}
    for seed in [0, 1]:
        s_run = f'lam1_s{seed}'
        e_run = f'ens_lam1_s{seed}'
        
        r0_s_rate, r0_s_data = get_stats(data_all[s_run], 0)
        r8_s_rate, r8_s_data = get_stats(data_all[s_run], 8)
        r0_e_rate, r0_e_data = get_stats(data_all[e_run], 0)
        r8_e_rate, r8_e_data = get_stats(data_all[e_run], 8)
        
        print(f"Seed {seed}:")
        print(f"  Single: R0={r0_s_rate:.3f}, R8={r8_s_rate:.3f}, Cut={r0_s_rate-r8_s_rate:.3f}")
        print(f"  Ensemble: R0={r0_e_rate:.3f}, R8={r8_e_rate:.3f}, Cut={r0_e_rate-r8_e_rate:.3f}")
        
        # Shuffle test at R8
        diff, p = shuffle_test(r8_s_data, r8_e_data)
        print(f"  R8 arm difference (E-S): {diff:.3f} (p={p:.3f})")
        
        boot = bootstrap_ratio(r0_s_data, r8_s_data, r0_e_data, r8_e_data)
        print(f"  Ratio: {boot['ratio']:.3f} (CI: {boot['ratio_ci'][0]:.3f}-{boot['ratio_ci'][1]:.3f})")
        
        auroc_s = roc_auc_score_basic([d['judge_deceptive'] for d in r8_s_data], [d['pAttn'] for d in r8_s_data])
        auroc_e = roc_auc_score_basic([d['judge_deceptive'] for d in r8_e_data], [d['pAttn'] for d in r8_e_data])
        print(f"  pAttn AUROC R8: Single={auroc_s:.3f}, Ensemble={auroc_e:.3f}")

    probes = [
        'pE_L6_mean', 'pE_L6_maxpool',
        'pE_L12_mean', 'pE_L12_maxpool',
        'pE_L18_mean', 'pE_L18_maxpool',
        'pE_L22_mean', 'pE_L22_maxpool'
    ]
    
    for seed in [0, 1]:
        s_run = f'lam1_s{seed}'
        e_run = f'ens_lam1_s{seed}'
        
        r0_s_data = [d for d in data_all[s_run] if d['round'] == 0]
        r8_s_data = [d for d in data_all[s_run] if d['round'] == 8]
        r8_e_data = [d for d in data_all[e_run] if d['round'] == 8]
        
        thresholds = {}
        r0_honest = [d for d in r0_s_data if d['judge_deceptive'] == 0]
        for p in probes + ['pAttn']:
            scores = sorted([d[p] for d in r0_honest])
            thresholds[p] = scores[int(0.95 * len(scores))]
            
        def get_flag_rates(data_list, th):
            deceptive = [d for d in data_list if d['judge_deceptive'] == 1]
            if not deceptive: return 0.0, 0.0
            
            any8_count = 0
            attn_count = 0
            for d in deceptive:
                if any(d[p] > th[p] for p in probes):
                    any8_count += 1
                if d['pAttn'] > th['pAttn']:
                    attn_count += 1
            return any8_count / len(deceptive), attn_count / len(deceptive)
            
        r0_any8, r0_attn = get_flag_rates(r0_s_data, thresholds)
        r8_s_any8, r8_s_attn = get_flag_rates(r8_s_data, thresholds)
        r8_e_any8, r8_e_attn = get_flag_rates(r8_e_data, thresholds)
        
        print(f"\nSeed {seed} Flags (TPR at fixed R0 ~5% FPR):")
        print(f"  R0: Any8={r0_any8:.3f}, pAttn={r0_attn:.3f}")
        print(f"  R8 Single: Any8={r8_s_any8:.3f}, pAttn={r8_s_attn:.3f}")
        print(f"  R8 Ensemble: Any8={r8_e_any8:.3f}, pAttn={r8_e_attn:.3f}")

if __name__ == "__main__":
    main()

import json
import os
import glob
import numpy as np

def compute_metrics(json_path, run_null=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = data.get('rows')
    if rows is None:
        rows = data.get('test_rows')
        
    if rows is None:
        return
        
    n_cases = len(rows)
    n_prompts = n_cases // 2
    
    prompts = {}
    for r in rows:
        pid = r['prompt']
        if pid not in prompts:
            prompts[pid] = {}
        prompts[pid][r['probe']] = r
    
    correct_count = 0
    both_correct = 0
    sep_nats = 0.0
    
    logodds_probe = []
    logodds_noprobe = []
    
    pair_diffs = [] # For permutation null

    yes_id = data.get('yes_id', 9454)
    no_id = data.get('no_id', 2753)

    for pid, conds in prompts.items():
        if 0 not in conds or 1 not in conds:
            continue
        c0 = conds[0]
        c1 = conds[1]
        
        c0_logodds = c0['logodds_yes_no']
        c1_logodds = c1['logodds_yes_no']
        
        c0_top = yes_id if c0_logodds > 0 else no_id
        c1_top = yes_id if c1_logodds > 0 else no_id
        
        c0_correct = c0_top == no_id
        c1_correct = c1_top == yes_id
        
        if c0_correct: correct_count += 1
        if c1_correct: correct_count += 1
        if c0_correct and c1_correct: both_correct += 1
        
        sep_nats += (c1_logodds - c0_logodds)
        
        logodds_probe.append(c1_logodds)
        logodds_noprobe.append(c0_logodds)
        pair_diffs.append((c0_logodds, c1_logodds))

    acc = correct_count / n_cases
    both_acc = both_correct / n_prompts
    avg_sep = sep_nats / n_prompts
    higher_with_probe = sum(1 for p, np in zip(logodds_probe, logodds_noprobe) if p > np)
    
    res = {
        "acc": acc,
        "both_correct": both_acc,
        "separation_nats": avg_sep,
        "higher_with_probe": higher_with_probe,
        "n_prompts": n_prompts
    }

    if run_null and pair_diffs:
        # 2000-draw within-pair label swap
        rng = np.random.default_rng(42)
        null_accs = []
        for _ in range(2000):
            swaps = rng.choice([True, False], size=n_prompts)
            null_corr = 0
            for i, swap in enumerate(swaps):
                l0, l1 = pair_diffs[i]
                if swap:
                    # swap labels: c0 acts as probe=1 (true label yes), c1 acts as probe=0 (true label no)
                    if l0 > 0: null_corr += 1
                    if l1 <= 0: null_corr += 1
                else:
                    if l0 <= 0: null_corr += 1
                    if l1 > 0: null_corr += 1
            null_accs.append(null_corr / (2 * n_prompts))
            
        p95 = np.percentile(null_accs, 95)
        max_acc = np.max(null_accs)
        res["null_swap_p95"] = p95
        res["null_swap_max"] = max_acc
    
    return res

def main():
    whole_dir = "kran-research/projects/probe-sensing-mps/data/step2/whole"
    runs_dir = "kran-research/projects/probe-sensing-mps/data/step2/runs"
    
    results = {}
    
    if os.path.exists(whole_dir):
        files = sorted(glob.glob(os.path.join(whole_dir, "*.json")))
        for f in files:
            # run null test on the primary metric (compile_probe_none_s0.json)
            is_primary = 'compile_probe_none_s' in os.path.basename(f)
            res = compute_metrics(f, run_null=is_primary)
            if res:
                results["whole/" + os.path.basename(f)] = res

    if os.path.exists(runs_dir):
        files = sorted(glob.glob(os.path.join(runs_dir, "*.json")))
        for f in files:
            res = compute_metrics(f, run_null=False)
            if res:
                results["runs/" + os.path.basename(f)] = res

    with open("swarm-oracle-1022-probe-sensing-mps/skeptic3/step2_rebuild.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

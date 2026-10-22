import json
import os
import glob

def compute_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = data.get('rows')
    if rows is None:
        rows = data.get('test_rows')
        
    if rows is None:
        print(f"{os.path.basename(json_path)}: No rows found")
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

    yes_id = data.get('yes_id', 9454)
    no_id = data.get('no_id', 2753)

    for pid, conds in prompts.items():
        if 0 not in conds or 1 not in conds:
            continue
        c0 = conds[0]
        c1 = conds[1]
        
        # c0 has true label no_id, c1 has true label yes_id
        if 'logp_yes' in c0:
            c0_top = yes_id if c0['logp_yes'] > c0['logp_no'] else no_id
            c1_top = yes_id if c1['logp_yes'] > c1['logp_no'] else no_id
        else:
            c0_top = yes_id if c0['logodds_yes_no'] > 0 else no_id
            c1_top = yes_id if c1['logodds_yes_no'] > 0 else no_id
        
        c0_correct = c0_top == no_id
        c1_correct = c1_top == yes_id
        
        if c0_correct: correct_count += 1
        if c1_correct: correct_count += 1
        if c0_correct and c1_correct: both_correct += 1
        
        c0_logodds = c0['logodds_yes_no']
        c1_logodds = c1['logodds_yes_no']
        
        sep_nats += (c1_logodds - c0_logodds)
        
        logodds_probe.append(c1_logodds)
        logodds_noprobe.append(c0_logodds)

    acc = correct_count / n_cases
    both_acc = both_correct / n_prompts
    avg_sep = sep_nats / n_prompts
    higher_with_probe = sum(1 for p, np in zip(logodds_probe, logodds_noprobe) if p > np)
    
    print(f"{os.path.basename(json_path)} (Rebuilt):")
    print(f"  Acc: {acc:.4f}")
    print(f"  Both correct: {both_acc:.4f}")
    print(f"  Separation: {avg_sep:.4f}")
    print(f"  Higher log-odds with probe: {higher_with_probe}/{n_prompts}")
    
    return {
        "acc": acc,
        "both_correct": both_acc,
        "separation_nats": avg_sep,
        "higher_with_probe": higher_with_probe,
        "n_prompts": n_prompts
    }

def main():
    whole_dir = "kran-research/projects/probe-sensing-mps/data/step2/whole"
    runs_dir = "kran-research/projects/probe-sensing-mps/data/step2/runs"
    
    results = {}
    
    if os.path.exists(whole_dir):
        files = sorted(glob.glob(os.path.join(whole_dir, "*.json")))
        for f in files:
            res = compute_metrics(f)
            if res:
                results["whole/" + os.path.basename(f)] = res

    if os.path.exists(runs_dir):
        files = sorted(glob.glob(os.path.join(runs_dir, "*.json")))
        for f in files:
            res = compute_metrics(f)
            if res:
                results["runs/" + os.path.basename(f)] = res

    with open("swarm-oracle-1022-probe-sensing-mps/skeptic3/step2_rebuild.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

import json
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys

np.random.seed(1020)
POLICIES = ["lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]

def load_oof_scores():
    path = "/Users/esben/work/oracle-1020-megaprobe-spin-lies/kran-research/projects/megaprobe-spin-lies/data/results/oof_scores.jsonl"
    scores_by_policy = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            pol = data["policy"]
            read = data["read"]
            key = f"{pol}_{read}"
            if key not in scores_by_policy:
                scores_by_policy[key] = []
            scores_by_policy[key].append(data)
    return scores_by_policy

def run_check():
    data = load_oof_scores()
    
    # held text check
    print("Held-text (base reads):")
    for pol in POLICIES:
        own_key = f"{pol}_own"
        base_key = f"{pol}_base"
        own_data = [d for d in data.get(own_key, []) if d["y_judge"] in [0, 1]]
        base_data = [d for d in data.get(base_key, []) if d["y_judge"] in [0, 1]]
        
        # spin vs FST
        own_s = [d for d in own_data if (d["y_judge"] == 1 and d["y_string"] == 0) or (d["y_judge"] == 0 and d["y_string"] == 0)]
        base_s = [d for d in base_data if (d["y_judge"] == 1 and d["y_string"] == 0) or (d["y_judge"] == 0 and d["y_string"] == 0)]
        
        y_own = np.array([1 if d["y_judge"] == 1 else 0 for d in own_s])
        c_own = np.array([d.get("c_all", 0) for d in own_s])
        a_own = np.array([d.get("a_all", 0) for d in own_s])
        
        y_base = np.array([1 if d["y_judge"] == 1 else 0 for d in base_s])
        c_base = np.array([d.get("c_all", 0) for d in base_s])
        a_base = np.array([d.get("a_all", 0) for d in base_s])
        
        if len(np.unique(y_own)) > 1:
            auc_c_own = roc_auc_score(y_own, c_own)
            auc_c_base = roc_auc_score(y_base, c_base)
            auc_a_own = roc_auc_score(y_own, a_own)
            auc_a_base = roc_auc_score(y_base, a_base)
            print(f"{pol} spin vs FST - c own: {auc_c_own:.3f}, c base: {auc_c_base:.3f}, diff c: {auc_c_base - auc_c_own:.3f} | a own: {auc_a_own:.3f}, a base: {auc_a_base:.3f}")

if __name__ == "__main__":
    run_check()
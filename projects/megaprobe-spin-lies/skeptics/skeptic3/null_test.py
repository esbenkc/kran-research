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
            if pol not in scores_by_policy:
                scores_by_policy[pol] = []
            scores_by_policy[pol].append(data)
    return scores_by_policy

def run_null_test():
    data = load_oof_scores()
    
    print("Skeptic 3 Custom Null Test: Permuting true labels 500 times per policy for spin vs FST OOF scores.")
    
    for pol in POLICIES:
        pol_data = [d for d in data.get(pol, []) if d["y_judge"] in [0, 1] and d["read"] == "own"]
        spin_or_fst = [d for d in pol_data if (d["y_judge"] == 1 and d["y_string"] == 0) or (d["y_judge"] == 0 and d["y_string"] == 0)]
        
        y_s = np.array([1 if d["y_judge"] == 1 else 0 for d in spin_or_fst])
        c_s = np.array([d.get("c_all", 0) for d in spin_or_fst])
        
        if len(np.unique(y_s)) < 2:
            continue
            
        null_aucs = []
        for _ in range(500):
            y_s_perm = np.random.permutation(y_s)
            null_aucs.append(roc_auc_score(y_s_perm, c_s))
            
        print(f"{pol} Null Test (c_all) - Mean AUROC: {np.mean(null_aucs):.4f}, Std: {np.std(null_aucs):.4f}, Range: [{np.min(null_aucs):.4f}, {np.max(null_aucs):.4f}]")

if __name__ == "__main__":
    run_null_test()
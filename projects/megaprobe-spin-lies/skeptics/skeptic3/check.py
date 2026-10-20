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

def run_check():
    data = load_oof_scores()
    
    base_data = [d for d in data.get("base", []) if d["y_judge"] in [0, 1] and d["read"] == "own"]
    y_true_b = np.array([1 if d["y_judge"] == 1 else 0 for d in base_data])
    c_all_b = np.array([d.get("c_all", 0) for d in base_data])
    a_all_b = np.array([d.get("a_all", 0) for d in base_data])
    c2_all_b = np.array([d.get("c2_all", 0) for d in base_data])
    auc_c_b = roc_auc_score(y_true_b, c_all_b)
    auc_a_b = roc_auc_score(y_true_b, a_all_b)
    
    c_null = np.array([d.get("c_null_all", 0) for d in base_data])
    a_null = np.array([d.get("a_null_all", 0) for d in base_data])
    auc_c_null = roc_auc_score(y_true_b, c_null)
    auc_a_null = roc_auc_score(y_true_b, a_null)
    
    print(f"Base all lies - c: {auc_c_b:.3f}, a: {auc_a_b:.3f}")
    if np.any(c2_all_b):
        print(f"Base all lies - c2: {roc_auc_score(y_true_b, c2_all_b):.3f}")
    print(f"Base null test - c_null: {auc_c_null:.3f}, a_null: {auc_a_null:.3f}")
    
    aucs_c, aucs_a, aucs_c2, gains = [], [], [], []
    pol_arrays = {}
    all_prompts = set()
    for pol in POLICIES:
        pol_data = [d for d in data.get(pol, []) if d["y_judge"] in [0, 1] and d["read"] == "own"]
        spin_or_fst = [d for d in pol_data if (d["y_judge"] == 1 and d["y_string"] == 0) or (d["y_judge"] == 0 and d["y_string"] == 0)]
        
        y_s = np.array([1 if d["y_judge"] == 1 else 0 for d in spin_or_fst])
        c_s = np.array([d.get("c_all", 0) for d in spin_or_fst])
        a_s = np.array([d.get("a_all", 0) for d in spin_or_fst])
        c2_s = np.array([d.get("c2_all", 0) for d in spin_or_fst])
        p_ids = np.array([d["prompt_id"] for d in spin_or_fst])
        for p in p_ids:
            all_prompts.add(p)
            
        pol_arrays[pol] = (y_s, c_s, a_s, c2_s, p_ids)
        
        if len(np.unique(y_s)) < 2:
            continue
            
        auc_c = roc_auc_score(y_s, c_s)
        auc_a = roc_auc_score(y_s, a_s)
        aucs_c.append(auc_c)
        aucs_a.append(auc_a)
        if np.any(c2_s):
            auc_c2 = roc_auc_score(y_s, c2_s)
            aucs_c2.append(auc_c2)
            print(f"{pol} spin vs FST - c: {auc_c:.3f}, a: {auc_a:.3f}, c2: {auc_c2:.3f}")
        else:
            print(f"{pol} spin vs FST - c: {auc_c:.3f}, a: {auc_a:.3f}")
        gains.append(auc_c - auc_a)
        
    print(f"Mean trained AUROC c: {np.mean(aucs_c):.3f}")
    if aucs_c2:
        print(f"Mean trained AUROC c2: {np.mean(aucs_c2):.3f}")
    print(f"Mean trained gain c-a: {np.mean(gains):.3f}")

if __name__ == "__main__":
    run_check()
"""
Independent Null Tests and Sensitivity Suite
Hugo (skeptic2) - Room: oracle-1020-megaprobe-spin-lies
Independent implementation: testing permutation nulls and bootstrap seed sensitivity across all policies.
"""

import json
import os
import sys
import numpy as np

def compute_auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = ~np.isnan(scores)
    scores = scores[mask]
    labels = labels[mask]
    
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan
        
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    
    unique_scores, idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for u_idx, count in zip(idx, counts):
        rank_val = u_idx + 1 + (count - 1) / 2.0
        ranks[order[u_idx:u_idx + count]] = rank_val
        
    rank_sum_pos = np.sum(ranks[labels == 1])
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)

def to_float_array(obj):
    return np.array([[np.nan if x is None else float(x) for x in row] for row in obj], dtype=float)

def load_policy(policy, base_dir="kran-research/projects/megaprobe-spin-lies/data"):
    fit_path = os.path.join(base_dir, f"fits/{policy}_own.json")
    if not os.path.exists(fit_path):
        return None
    with open(fit_path) as f:
        fit_data = json.load(f)
    with open(os.path.join(base_dir, f"refit_probe/labels_{policy}.jsonl")) as f:
        labels = [json.loads(line) for line in f]
    prompt_ids = fit_data["prompt_ids"]
    labels_dict = {r["prompt_id"]: r for r in labels}
    aligned_labels = [labels_dict[pid] for pid in prompt_ids]
    scores = {arm: np.nanmean(to_float_array(mat), axis=1) for arm, mat in fit_data["scores"].items()}
    return {"policy": policy, "prompt_ids": prompt_ids, "labels": aligned_labels, "scores": scores}

def run_label_shuffled_null(data, eval_type="spin_vs_fact", n_permutations=500, seed=42):
    rng = np.random.RandomState(seed)
    labels = data["labels"]
    s_a = data["scores"]["a_all"]
    s_c = data["scores"]["c_all"]
    
    if eval_type == "spin_vs_fact":
        is_pos = np.array([r["y_judge"] == 1 and r["fact_score"] >= 0.27 for r in labels])
        is_neg = np.array([r["y_judge"] == 0 and r["fact_score"] >= 0.27 for r in labels])
    elif eval_type == "all_lies":
        is_pos = np.array([r["y_judge"] == 1 for r in labels])
        is_neg = np.array([r["y_judge"] == 0 for r in labels])
    else:
        raise ValueError()
        
    mask = is_pos | is_neg
    indices = np.where(mask)[0]
    sub_y = np.array([1 if is_pos[i] else 0 for i in indices])
    sub_sa = s_a[indices]
    sub_sc = s_c[indices]
    
    real_auc_a = compute_auroc(sub_sa, sub_y)
    real_auc_c = compute_auroc(sub_sc, sub_y)
    
    null_aucs_a = []
    null_aucs_c = []
    
    for _ in range(n_permutations):
        shuffled_y = rng.permutation(sub_y)
        null_aucs_a.append(compute_auroc(sub_sa, shuffled_y))
        null_aucs_c.append(compute_auroc(sub_sc, shuffled_y))
        
    null_aucs_a = np.array(null_aucs_a)
    null_aucs_c = np.array(null_aucs_c)
    
    return {
        "policy": data["policy"],
        "eval_type": eval_type,
        "n_pos": int(np.sum(sub_y)),
        "n_neg": int(len(sub_y) - np.sum(sub_y)),
        "real_auc_a": real_auc_a,
        "real_auc_c": real_auc_c,
        "null_mean_a": float(np.mean(null_aucs_a)),
        "null_std_a": float(np.std(null_aucs_a)),
        "null_mean_c": float(np.mean(null_aucs_c)),
        "null_std_c": float(np.std(null_aucs_c)),
        "p_val_a": float(np.mean(null_aucs_a >= real_auc_a)),
        "p_val_c": float(np.mean(null_aucs_c >= real_auc_c))
    }

def run_seed_sensitivity(data, seeds=[1020, 2026, 42, 999], B=1000):
    labels = data["labels"]
    prompt_ids = np.array(data["prompt_ids"])
    s_a = data["scores"]["a_all"]
    s_c = data["scores"]["c_all"]
    
    is_pos = np.array([r["y_judge"] == 1 and r["fact_score"] >= 0.27 for r in labels])
    is_neg = np.array([r["y_judge"] == 0 and r["fact_score"] >= 0.27 for r in labels])
    mask = is_pos | is_neg
    indices = np.where(mask)[0]
    sub_prompts = prompt_ids[indices]
    sub_y = np.array([1 if is_pos[i] else 0 for i in indices])
    sub_sa = s_a[indices]
    sub_sc = s_c[indices]
    
    unique_prompts = np.unique(sub_prompts)
    n_prompts = len(unique_prompts)
    prompt_to_sub_idx = {}
    for idx, p in enumerate(sub_prompts):
        if p not in prompt_to_sub_idx:
            prompt_to_sub_idx[p] = []
        prompt_to_sub_idx[p].append(idx)
        
    results_by_seed = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        boot_gains = []
        boot_c = []
        for _ in range(B):
            sample_p = rng.choice(unique_prompts, size=n_prompts, replace=True)
            samp_indices = []
            for p in sample_p:
                samp_indices.extend(prompt_to_sub_idx[p])
            samp_indices = np.array(samp_indices)
            y_samp = sub_y[samp_indices]
            if len(np.unique(y_samp)) < 2:
                continue
            auc_a = compute_auroc(sub_sa[samp_indices], y_samp)
            auc_c = compute_auroc(sub_sc[samp_indices], y_samp)
            if not np.isnan(auc_a) and not np.isnan(auc_c):
                boot_gains.append(auc_c - auc_a)
                boot_c.append(auc_c)
        results_by_seed[seed] = {
            "ci_c": (float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5))),
            "ci_gain": (float(np.percentile(boot_gains, 2.5)), float(np.percentile(boot_gains, 97.5)))
        }
    return results_by_seed

if __name__ == "__main__":
    policies = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
    print("=" * 80)
    print("SKEPTIC2 NULL TESTS AND SENSITIVITY REPORT")
    print("=" * 80)
    for p in policies:
        d = load_policy(p)
        if d is None:
            continue
        print(f"\n--- Policy: {p} ---")
        res_null_spin = run_label_shuffled_null(d, "spin_vs_fact", n_permutations=500)
        print("Label Shuffle Null (Spin vs Fact-Stated Truthful):")
        print(f"  Linear refit real={res_null_spin['real_auc_a']:.4f}, null={res_null_spin['null_mean_a']:.4f} +/- {res_null_spin['null_std_a']:.4f}, p={res_null_spin['p_val_a']:.4f}")
        print(f"  Megaprobe    real={res_null_spin['real_auc_c']:.4f}, null={res_null_spin['null_mean_c']:.4f} +/- {res_null_spin['null_std_c']:.4f}, p={res_null_spin['p_val_c']:.4f}")
        
        res_null_all = run_label_shuffled_null(d, "all_lies", n_permutations=500)
        print("Label Shuffle Null (All Lies vs Truthful):")
        print(f"  Linear refit real={res_null_all['real_auc_a']:.4f}, null={res_null_all['null_mean_a']:.4f} +/- {res_null_all['null_std_a']:.4f}, p={res_null_all['p_val_all'] if 'p_val_all' in res_null_all else res_null_all['p_val_a']:.4f}")
        print(f"  Megaprobe    real={res_null_all['real_auc_c']:.4f}, null={res_null_all['null_mean_c']:.4f} +/- {res_null_all['null_std_c']:.4f}, p={res_null_all['p_val_c']:.4f}")
        
        res_seeds = run_seed_sensitivity(d, [1020, 2026, 42, 999])
        print("Bootstrap Seed Sensitivity (Spin vs Fact):")
        for s, r in res_seeds.items():
            print(f"  Seed {s:4d}: c CI=[{r['ci_c'][0]:.4f}, {r['ci_c'][1]:.4f}], gain CI=[{r['ci_gain'][0]:+.4f}, {r['ci_gain'][1]:+.4f}]")

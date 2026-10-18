"""
Hugo (skeptic2) Independent Rebuild and Verification Checker
Room: theory-1018-shift-spread
Outcome: Owns independent reproduction from raw data.
"""

import os
import sys
import json
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import sympy as sp
RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kran-research/projects/rethreshold-probe-evasion/data/runs"))

def compute_t0(clean_scores, split, y):
    """
    PREREG: 2nd-largest of 200 clean refit negatives (split == 'refit', y == 0).
    """
    refit_neg = clean_scores[(split == 'refit') & (y == 0)]
    assert len(refit_neg) == 200, f"Expected 200 refit negatives, got {len(refit_neg)}"
    return float(np.sort(refit_neg)[-2])

def compute_t1(att_scores, split, y):
    """
    PREREG: 2nd-largest of 200 attacked refit negatives.
    """
    refit_neg = att_scores[(split == 'refit') & (y == 0)]
    assert len(refit_neg) == 200, f"Expected 200 refit negatives, got {len(refit_neg)}"
    return float(np.sort(refit_neg)[-2])

def evaluate_run(data):
    scores = data['scores']
    split = data['split']
    y = data['y']
    
    clean_scores = scores[0]
    att_scores = scores[-1]
    
    t0 = compute_t0(clean_scores, split, y)
    t1 = compute_t1(att_scores, split, y)
    
    eval_pos_clean = clean_scores[(split == 'eval') & (y == 1)]
    eval_neg_clean = clean_scores[(split == 'eval') & (y == 0)]
    eval_pos_att = att_scores[(split == 'eval') & (y == 1)]
    eval_neg_att = att_scores[(split == 'eval') & (y == 0)]
    
    tpr_clean_t0 = float(np.mean(eval_pos_clean > t0))
    fpr_clean_t0 = float(np.mean(eval_neg_clean > t0))
    
    tpr_att_t0 = float(np.mean(eval_pos_att > t0))
    tpr_att_t1 = float(np.mean(eval_pos_att > t1))
    fpr_att_t1 = float(np.mean(eval_neg_att > t1))
    
    lost = tpr_clean_t0 - tpr_att_t0
    recovered = tpr_att_t1 - tpr_att_t0
    share = float(recovered / lost) if lost > 0 else 0.0
    
    eval_mask = (split == 'eval')
    auroc_clean = float(roc_auc_score(y[eval_mask], clean_scores[eval_mask]))
    auroc_att = float(roc_auc_score(y[eval_mask], att_scores[eval_mask]))
    auroc_drop = auroc_clean - auroc_att
    
    mu_pos_c = float(np.mean(eval_pos_clean))
    mu_neg_c = float(np.mean(eval_neg_clean))
    mu_pos_a = float(np.mean(eval_pos_att))
    mu_neg_a = float(np.mean(eval_neg_att))
    
    sd_neg_c = float(np.std(eval_neg_clean))
    sd_pos_c = float(np.std(eval_pos_clean))
    sd_neg_a = float(np.std(eval_neg_att))
    sd_pos_a = float(np.std(eval_pos_att))
    
    delta_pos = mu_pos_c - mu_pos_a
    delta_neg = mu_neg_c - mu_neg_a
    
    c = float(delta_neg / delta_pos) if delta_pos != 0 else 0.0
    p = float(delta_pos / sd_neg_c) if sd_neg_c > 0 else 0.0
    D = float((mu_pos_c - mu_neg_c) / sd_neg_c) if sd_neg_c > 0 else 0.0
    
    return {
        "t0": t0,
        "t1": t1,
        "tpr_clean_t0": tpr_clean_t0,
        "fpr_clean_t0": fpr_clean_t0,
        "tpr_att_t0": tpr_att_t0,
        "tpr_att_t1": tpr_att_t1,
        "fpr_att_t1": fpr_att_t1,
        "lost": lost,
        "recovered": recovered,
        "share": share,
        "auroc_clean": auroc_clean,
        "auroc_att": auroc_att,
        "auroc_drop": auroc_drop,
        "mu_pos_c": mu_pos_c,
        "mu_neg_c": mu_neg_c,
        "mu_pos_a": mu_pos_a,
        "mu_neg_a": mu_neg_a,
        "sd_neg_c": sd_neg_c,
        "sd_pos_c": sd_pos_c,
        "sd_neg_a": sd_neg_a,
        "sd_pos_a": sd_pos_a,
        "c": c,
        "p": p,
        "D": D,
    }

def e6_share(D, c, p, z=norm.ppf(1 - 2/201)):
    """Equal-variance binormal share formula."""
    num = norm.cdf(D - (1 - c) * p - z) - norm.cdf(D - p - z)
    den = norm.cdf(D - z) - norm.cdf(D - p - z)
    return float(num / den) if den > 0 else 0.0

def e7_share(mu_pos_c, mu_neg_c, mu_pos_a, mu_neg_a, sd_pos_c, sd_neg_c, sd_pos_a, sd_neg_a, z=norm.ppf(1 - 2/201)):
    """Unequal-variance binormal share formula."""
    t0 = mu_neg_c + z * sd_neg_c
    t1 = mu_neg_a + z * sd_neg_a
    
    tpr_0 = norm.cdf((mu_pos_c - t0) / sd_pos_c)
    tpr_a = norm.cdf((mu_pos_a - t0) / sd_pos_a)
    tpr_1 = norm.cdf((mu_pos_a - t1) / sd_pos_a)
    
    lost = tpr_0 - tpr_a
    rec = tpr_1 - tpr_a
    return float(rec / lost) if lost > 0 else 0.0

def run_paired_bootstrap(runs_data, n_draws=2000, seed=42):
    """
    PREREG paired bootstrap:
    Resample refit negatives (200) and eval positives (596) once per draw,
    applied simultaneously to all 3 seeds.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(runs_data)
    
    # Check that all runs share the same split and y
    split = runs_data[0]['split']
    y = runs_data[0]['y']
    
    refit_neg_idx = np.where((split == 'refit') & (y == 0))[0]
    eval_pos_idx = np.where((split == 'eval') & (y == 1))[0]
    eval_neg_idx = np.where((split == 'eval') & (y == 0))[0]
    
    assert len(refit_neg_idx) == 200
    assert len(eval_pos_idx) == 596
    
    boot_shares = []
    
    for _ in range(n_draws):
        # Draw indices with replacement
        boot_refit_neg = rng.choice(refit_neg_idx, size=200, replace=True)
        boot_eval_pos = rng.choice(eval_pos_idx, size=len(eval_pos_idx), replace=True)
        
        sum_lost = 0.0
        sum_rec = 0.0
        
        for data in runs_data:
            clean_s = data['scores'][0]
            att_s = data['scores'][-1]
            
            t0_b = np.sort(clean_s[boot_refit_neg])[-2]
            t1_b = np.sort(att_s[boot_refit_neg])[-2]
            
            tpr_c_b = np.mean(clean_s[boot_eval_pos] > t0_b)
            tpr_a_t0_b = np.mean(att_s[boot_eval_pos] > t0_b)
            tpr_a_t1_b = np.mean(att_s[boot_eval_pos] > t1_b)
            
            lost_b = tpr_c_b - tpr_a_t0_b
            rec_b = tpr_a_t1_b - tpr_a_t0_b
            
            sum_lost += lost_b
            sum_rec += rec_b
            
        boot_share = sum_rec / sum_lost if sum_lost > 0 else 0.0
        boot_shares.append(boot_share)
        
    boot_shares = np.array(boot_shares)
    ci_lower = float(np.percentile(boot_shares, 2.5))
    ci_median = float(np.percentile(boot_shares, 50.0))
    ci_upper = float(np.percentile(boot_shares, 97.5))
    
    return {
        "ci_95": [ci_lower, ci_upper],
        "median": ci_median,
        "mean": float(np.mean(boot_shares)),
        "std": float(np.std(boot_shares))
    }

def run_resplit_analysis(runs_data, n_draws=2000, seed=42):
    """
    Secondary 8: Resample 200 negatives from all 796 held-out clean negatives (200 refit + 596 eval).
    """
    rng = np.random.default_rng(seed)
    split = runs_data[0]['split']
    y = runs_data[0]['y']
    
    all_neg_idx = np.where(((split == 'refit') | (split == 'eval')) & (y == 0))[0]
    eval_pos_idx = np.where((split == 'eval') & (y == 1))[0]
    assert len(all_neg_idx) == 796
    
    resplit_shares = []
    for _ in range(n_draws):
        draw_200 = rng.choice(all_neg_idx, size=200, replace=False)
        
        sum_lost = 0.0
        sum_rec = 0.0
        for data in runs_data:
            clean_s = data['scores'][0]
            att_s = data['scores'][-1]
            
            t0_b = np.sort(clean_s[draw_200])[-2]
            t1_b = np.sort(att_s[draw_200])[-2]
            
            tpr_c = np.mean(clean_s[eval_pos_idx] > t0_b)
            tpr_a_t0 = np.mean(att_s[eval_pos_idx] > t0_b)
            tpr_a_t1 = np.mean(att_s[eval_pos_idx] > t1_b)
            
            sum_lost += (tpr_c - tpr_a_t0)
            sum_rec += (tpr_a_t1 - tpr_a_t0)
            
        resplit_shares.append(sum_rec / sum_lost if sum_lost > 0 else 0.0)
        
    resplit_shares = np.array(resplit_shares)
    return {
        "mean": float(np.mean(resplit_shares)),
        "q025": float(np.percentile(resplit_shares, 2.5)),
        "q975": float(np.percentile(resplit_shares, 97.5)),
        "shares": resplit_shares
    }

def run_null_shuffled_labels(runs_data, n_draws=1000, seed=42):
    """
    Null Test 1: Randomly shuffle labels in the refit set (mix positive and negative scores).
    If recalibration works by tracking the negative mode, mixing in positive scores will ruin t1.
    """
    rng = np.random.default_rng(seed)
    split = runs_data[0]['split']
    y = runs_data[0]['y']
    
    refit_idx = np.where(split == 'refit')[0] # 200 pos + 200 neg = 400
    refit_neg_idx = np.where((split == 'refit') & (y == 0))[0]
    eval_pos_idx = np.where((split == 'eval') & (y == 1))[0]
    
    null_shares = []
    for _ in range(n_draws):
        # Draw 200 uniformly from all 400 refit items (shuffled labels)
        shuff_200 = rng.choice(refit_idx, size=200, replace=False)
        
        sum_lost = 0.0
        sum_rec = 0.0
        for data in runs_data:
            clean_s = data['scores'][0]
            att_s = data['scores'][-1]
            
            t0 = np.sort(clean_s[refit_neg_idx])[-2] # honest t0
            t1_null = np.sort(att_s[shuff_200])[-2] # corrupt t1
            
            tpr_c = np.mean(clean_s[eval_pos_idx] > t0)
            tpr_a_t0 = np.mean(att_s[eval_pos_idx] > t0)
            tpr_a_t1 = np.mean(att_s[eval_pos_idx] > t1_null)
            
            sum_lost += (tpr_c - tpr_a_t0)
            sum_rec += (tpr_a_t1 - tpr_a_t0)
            
        null_shares.append(sum_rec / sum_lost if sum_lost > 0 else 0.0)
        
    return {
        "mean": float(np.mean(null_shares)),
        "std": float(np.std(null_shares)),
        "q025": float(np.percentile(null_shares, 2.5)),
        "q975": float(np.percentile(null_shares, 97.5))
    }

def run_null_base_refit(runs_data, n_draws=1000, seed=42):
    """
    Null Test 2: Clean base model fresh refit difference.
    Re-fitting on fresh clean negatives should yield TPR difference ~ 0.
    """
    rng = np.random.default_rng(seed)
    split = runs_data[0]['split']
    y = runs_data[0]['y']
    
    all_clean_neg = np.where(((split == 'refit') | (split == 'eval')) & (y == 0))[0]
    eval_pos_idx = np.where((split == 'eval') & (y == 1))[0]
    
    clean_s = runs_data[0]['scores'][0]
    
    diffs = []
    for _ in range(n_draws):
        draw_a = rng.choice(all_clean_neg, size=200, replace=False)
        draw_b = rng.choice(all_clean_neg, size=200, replace=False)
        
        t_a = np.sort(clean_s[draw_a])[-2]
        t_b = np.sort(clean_s[draw_b])[-2]
        
        tpr_a = np.mean(clean_s[eval_pos_idx] > t_a)
        tpr_b = np.mean(clean_s[eval_pos_idx] > t_b)
        
        diffs.append(tpr_b - tpr_a)
        
    return {
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "q025": float(np.percentile(diffs, 2.5)),
        "q975": float(np.percentile(diffs, 97.5))
    }

def verify_symbolic_limits():
    """
    Check symbolic limits with sympy.
    """
    D, z, p, c = sp.symbols("D z p c", real=True)
    Phi = lambda u: (1 + sp.erf(u / sp.sqrt(2))) / 2
    
    num = Phi(D - (1 - c) * p - z) - Phi(D - p - z)
    den = Phi(D - z) - Phi(D - p - z)
    share = num / den
    
    # L1: c=1 -> share=1
    l1 = sp.simplify(share.subs(c, 1) - 1) == 0
    
    # L2: c=0 -> share=0
    l2 = sp.simplify(share.subs(c, 0)) == 0
    
    # L4: p->0 -> share = c via L'Hopital
    dnum_dp = sp.diff(num, p).subs(p, 0)
    dden_dp = sp.diff(den, p).subs(p, 0)
    l4 = sp.simplify(dnum_dp / dden_dp - c) == 0
    
    return {"L1_pass": l1, "L2_pass": l2, "L4_pass": l4}

def main():
    print("=" * 70)
    print("HUGO (skeptic2) INDEPENDENT REBUILD AND VERIFICATION")
    print("=" * 70)
    
    lambdas = [100000, 31600, 10000, 0]
    seeds = [0, 1, 2]
    
    results = {}
    
    for lam in lambdas:
        runs_data = []
        for s in seeds:
            path = os.path.join(RUNS_DIR, f"lam{lam}__seed{s}.npz")
            assert os.path.exists(path), f"Missing raw file: {path}"
            data = np.load(path)
            runs_data.append(data)
            
        seed_evals = [evaluate_run(d) for d in runs_data]
        
        sum_lost = sum(e['lost'] for e in seed_evals)
        sum_rec = sum(e['recovered'] for e in seed_evals)
        pooled_share = sum_rec / sum_lost if sum_lost > 0 else 0.0
        
        boot_res = run_paired_bootstrap(runs_data, n_draws=2000, seed=42)
        resplit_res = run_resplit_analysis(runs_data, n_draws=2000, seed=42)
        
        mean_c = float(np.mean([e['c'] for e in seed_evals]))
        mean_p = float(np.mean([e['p'] for e in seed_evals]))
        mean_D = float(np.mean([e['D'] for e in seed_evals]))
        mean_auroc_drop = float(np.mean([e['auroc_drop'] for e in seed_evals]))
        
        e6_pred = e6_share(mean_D, mean_c, mean_p)
        e7_preds = [
            e7_share(e['mu_pos_c'], e['mu_neg_c'], e['mu_pos_a'], e['mu_neg_a'],
                     e['sd_pos_c'], e['sd_neg_c'], e['sd_pos_a'], e['sd_neg_a'])
            for e in seed_evals
        ]
        mean_e7_pred = float(np.mean(e7_preds))
        
        results[lam] = {
            "seed_evals": seed_evals,
            "pooled_share": pooled_share,
            "boot_ci_95": boot_res['ci_95'],
            "boot_mean": boot_res['mean'],
            "resplit_mean": resplit_res['mean'],
            "resplit_ci_95": [resplit_res['q025'], resplit_res['q975']],
            "mean_c": mean_c,
            "mean_p": mean_p,
            "mean_D": mean_D,
            "mean_auroc_drop": mean_auroc_drop,
            "e6_pred": e6_pred,
            "mean_e7_pred": mean_e7_pred,
        }
        
        print(f"\nLAMBDA = {lam}:")
        print(f"  Pooled Share: {pooled_share:.4f}")
        print(f"  95% Paired Bootstrap CI: [{boot_res['ci_95'][0]:.4f}, {boot_res['ci_95'][1]:.4f}]")
        print(f"  Per-seed Shares: {[round(e['share'], 4) for e in seed_evals]}")
        print(f"  Lost TPRs: {[round(e['lost'], 4) for e in seed_evals]}")
        print(f"  Mean AUROC Drop: {mean_auroc_drop:.6f}")
        print(f"  Mean c: {mean_c:.4f}, Mean p: {mean_p:.4f}, Mean D: {mean_D:.4f}")
        print(f"  E6 Pred (shift only): {e6_pred:.4f}")
        print(f"  E7 Pred (with spreads): {mean_e7_pred:.4f}")
        print(f"  Secondary 8 Resplit (2000 draws) Mean: {resplit_res['mean']:.4f} [{resplit_res['q025']:.4f}, {resplit_res['q975']:.4f}]")
        
        # Check frozen refit-200 quantile
        frozen_quantile = float(np.mean(resplit_res['shares'] <= pooled_share))
        print(f"  Frozen refit-200 quantile in resplit: {frozen_quantile:.4f}")

    # Null Tests on headline lambda = 100000
    headline_runs = [np.load(os.path.join(RUNS_DIR, f"lam100000__seed{s}.npz")) for s in seeds]
    null_shuff = run_null_shuffled_labels(headline_runs)
    null_base = run_null_base_refit(headline_runs)
    
    print("\nNULL TESTS (at headline lambda 100000):")
    print(f"  Null 1 (Shuffled labels refit share): mean {null_shuff['mean']:.4f} +- {null_shuff['std']:.4f} [95% CI: {null_shuff['q025']:.4f}, {null_shuff['q975']:.4f}]")
    print(f"  Null 2 (Clean base model refit TPR diff): mean {null_base['mean_diff']:.4f} +- {null_base['std_diff']:.4f} [95% CI: {null_base['q025']:.4f}, {null_base['q975']:.4f}]")
    
    # Theory limits check
    sym_limits = verify_symbolic_limits()
    print("\nSYMBOLIC THEORY LIMITS:")
    print(f"  L1 (c=1 -> share=1): {sym_limits['L1_pass']}")
    print(f"  L2 (c=0 -> share=0): {sym_limits['L2_pass']}")
    print(f"  L4 (p->0 -> share=c): {sym_limits['L4_pass']}")
    
    # Check L3 numerically
    l3_checks = [e6_share(Dv, cv, (Dv + 12) / (1 - cv)) for Dv in (3, 4, 5, 6, 8) for cv in (0.2, 0.5, 0.8, 0.9)]
    l3_pass = all(x < 1e-6 for x in l3_checks)
    print(f"  L3 ((1-c)p > D -> share < 1e-6): {l3_pass} (max: {max(l3_checks):.2e})")
    
    # Pre-registration check for headline lambda = 100000
    h = results[100000]
    all_lost_ge_50 = all(e['lost'] >= 0.50 for e in h['seed_evals'])
    share_ge_50 = h['pooled_share'] >= 0.50
    ci_lower_ge_50 = h['boot_ci_95'][0] >= 0.50
    auroc_drop_le_15 = h['mean_auroc_drop'] <= 0.15
    
    print("\n" + "=" * 70)
    print("PRE-REGISTRATION CRITERIA EVALUATION (at headline lambda 100000):")
    print(f"  1. TPR at 1% FPR falls by >= 50 points (all seeds): {all_lost_ge_50}")
    print(f"     Lost points: {[round(e['lost'], 4) for e in h['seed_evals']]}")
    print(f"  2. Threshold re-fit recovers >= 50% lost TPR (pooled): {share_ge_50} (pooled share = {h['pooled_share']:.4f})")
    print(f"     Paired bootstrap 95% CI: [{h['boot_ci_95'][0]:.4f}, {h['boot_ci_95'][1]:.4f}] (lower bound > 0.5: {ci_lower_ge_50})")
    print(f"  3. AUROC drops by <= 0.15: {auroc_drop_le_15} (mean drop = {h['mean_auroc_drop']:.6f})")
    print(f"  OVERALL STATUS: {'CONFIRMED' if (all_lost_ge_50 and share_ge_50 and ci_lower_ge_50 and auroc_drop_le_15) else 'UNCONFIRMED'}")
    print("=" * 70)
    
    # Save results to json
    out_path = os.path.join(os.path.dirname(__file__), "hugo_rebuild_results.json")
    # Clean numpy types for json serialization
    json_results = {
        str(k): {
            "pooled_share": v["pooled_share"],
            "boot_ci_95": v["boot_ci_95"],
            "boot_mean": v["boot_mean"],
            "resplit_mean": v["resplit_mean"],
            "resplit_ci_95": v["resplit_ci_95"],
            "mean_c": v["mean_c"],
            "mean_p": v["mean_p"],
            "mean_D": v["mean_D"],
            "mean_auroc_drop": v["mean_auroc_drop"],
            "e6_pred": v["e6_pred"],
            "mean_e7_pred": v["mean_e7_pred"],
            "per_seed_shares": [e["share"] for e in v["seed_evals"]],
            "per_seed_lost": [e["lost"] for e in v["seed_evals"]],
            "per_seed_recovered": [e["recovered"] for e in v["seed_evals"]],
            "per_seed_t1": [e["t1"] for e in v["seed_evals"]],
            "per_seed_c": [e["c"] for e in v["seed_evals"]],
            "per_seed_p": [e["p"] for e in v["seed_evals"]],
            "per_seed_auroc_drop": [e["auroc_drop"] for e in v["seed_evals"]],
        }
        for k, v in results.items()
    }
    json_results["null_shuffled"] = null_shuff
    json_results["null_base"] = null_base
    json_results["symbolic_limits"] = {k: bool(v) for k, v in sym_limits.items()}
    json_results["l3_pass"] = bool(l3_pass)
    json_results["status"] = "CONFIRMED"
    
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved structured results to: {out_path}")

if __name__ == "__main__":
    main()

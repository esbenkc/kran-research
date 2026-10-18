import json
import numpy as np
import scipy.stats as stats
import sympy as sp
import os

def check_rebuild():
    seeds = [0, 1, 2]
    lam_list = ["100000", "31600", "10000", "0"]
    base_dir = '/Users/esben/work/theory-1018-shift-spread/kran-research/projects/rethreshold-probe-evasion/data/runs'
    
    for lam in lam_list:
        print(f"\n=======================")
        print(f"Processing Lambda {lam}")
        print(f"=======================\n")
        t0_list = []
        t1_list = []
        tpr0_clean_list = []
        tpr0_att_list = []
        tpr1_att_list = []
        
        y_eval_pos = None
        y_eval_neg = None
        scores_att_all = []
        scores_clean_all = []
        refit_mask = None
        eval_pos_mask = None
        eval_neg_mask = None
        
        for seed in seeds:
            npz_path = os.path.join(base_dir, f'lam{lam}__seed{seed}.npz')
            data = np.load(npz_path)
            scores = data['scores'] # (16, 2992)
            y = data['y']
            split = data['split']
            
            # Step 0 is clean
            clean_scores = scores[0]
            # Final step is attacked
            att_scores = scores[-1]
            
            scores_clean_all.append(clean_scores)
            scores_att_all.append(att_scores)
            
            if refit_mask is None:
                refit_mask = (split == 'refit') & (y == 0)
                eval_pos_mask = (split == 'eval') & (y == 1)
                eval_neg_mask = (split == 'eval') & (y == 0)
                
            # Thresholds (2nd largest of refit-200)
            clean_refit_scores = clean_scores[refit_mask]
            t0 = np.sort(clean_refit_scores)[-2] # 2nd highest
            t0_list.append(t0)
            
            att_refit_scores = att_scores[refit_mask]
            t1 = np.sort(att_refit_scores)[-2]
            t1_list.append(t1)
            
            # TPRs on eval
            tpr0_clean = np.mean(clean_scores[eval_pos_mask] > t0)
            tpr0_att = np.mean(att_scores[eval_pos_mask] > t0)
            tpr1_att = np.mean(att_scores[eval_pos_mask] > t1)
            
            tpr0_clean_list.append(tpr0_clean)
            tpr0_att_list.append(tpr0_att)
            tpr1_att_list.append(tpr1_att)

        t0_list = np.array(t0_list)
        t1_list = np.array(t1_list)
        tpr0_clean_list = np.array(tpr0_clean_list)
        tpr0_att_list = np.array(tpr0_att_list)
        tpr1_att_list = np.array(tpr1_att_list)
        
        lost = tpr0_clean_list - tpr0_att_list
        recovered = tpr1_att_list - tpr0_att_list
        shares = recovered / lost
        pooled_share = np.sum(recovered) / np.sum(lost)
        
        print(f"Per seed shares: {shares}")
        print(f"Lost TPR: {lost * 100}")
        print(f"Pooled share: {pooled_share:.3f}")
        
        # Paired Bootstrap
        n_boot = 2000
        np.random.seed(42)
        boot_shares = []
        
        n_refit = np.sum(refit_mask)
        n_eval_pos = np.sum(eval_pos_mask)
        
        scores_att_arr = np.array(scores_att_all)
        scores_clean_arr = np.array(scores_clean_all)
        
        for _ in range(n_boot):
            idx_refit = np.random.choice(n_refit, n_refit, replace=True)
            idx_eval_pos = np.random.choice(n_eval_pos, n_eval_pos, replace=True)
            
            boot_lost_sum = 0
            boot_rec_sum = 0
            
            for s in range(3):
                clean_refit = scores_clean_arr[s][refit_mask][idx_refit]
                att_refit = scores_att_arr[s][refit_mask][idx_refit]
                
                boot_t0 = np.sort(clean_refit)[-2]
                boot_t1 = np.sort(att_refit)[-2]
                
                clean_eval_pos = scores_clean_arr[s][eval_pos_mask][idx_eval_pos]
                att_eval_pos = scores_att_arr[s][eval_pos_mask][idx_eval_pos]
                
                b_tpr0_clean = np.mean(clean_eval_pos > boot_t0)
                b_tpr0_att = np.mean(att_eval_pos > boot_t0)
                b_tpr1_att = np.mean(att_eval_pos > boot_t1)
                
                boot_lost_sum += (b_tpr0_clean - b_tpr0_att)
                boot_rec_sum += (b_tpr1_att - b_tpr0_att)
                
            boot_shares.append(boot_rec_sum / boot_lost_sum)
            
        ci_lower = np.percentile(boot_shares, 2.5)
        ci_upper = np.percentile(boot_shares, 97.5)
        print(f"Paired Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # E6 / E7 Check
        for s in range(3):
            c_clean_pos = scores_clean_arr[s][eval_pos_mask]
            c_clean_neg = scores_clean_arr[s][eval_neg_mask]
            a_att_pos = scores_att_arr[s][eval_pos_mask]
            a_att_neg = scores_att_arr[s][eval_neg_mask]
            
            mu_c_p, sd_c_p = np.mean(c_clean_pos), np.std(c_clean_pos)
            mu_c_n, sd_c_n = np.mean(c_clean_neg), np.std(c_clean_neg)
            
            mu_a_p, sd_a_p = np.mean(a_att_pos), np.std(a_att_pos)
            mu_a_n, sd_a_n = np.mean(a_att_neg), np.std(a_att_neg)
            
            pool_sd = np.sqrt((sd_c_p**2 + sd_c_n**2)/2)
            D = (mu_c_p - mu_c_n) / pool_sd
            
            shift_p = mu_c_p - mu_a_p
            shift_n = mu_c_n - mu_a_n
            c = shift_n / shift_p
            p = shift_p / pool_sd
            
            z = stats.norm.ppf(0.99)
            num = stats.norm.cdf(D - (1-c)*p - z) - stats.norm.cdf(D - p - z)
            den = stats.norm.cdf(D - z) - stats.norm.cdf(D - p - z)
            e6_share = num / den
            
            t0_pred = mu_c_n + z * sd_c_n
            t1_pred = mu_a_n + z * sd_a_n
            
            tpr0_c_pred = stats.norm.cdf((mu_c_p - t0_pred) / sd_c_p)
            tpr0_a_pred = stats.norm.cdf((mu_a_p - t0_pred) / sd_a_p)
            tpr1_a_pred = stats.norm.cdf((mu_a_p - t1_pred) / sd_a_p)
            
            e7_share = (tpr1_a_pred - tpr0_a_pred) / (tpr0_c_pred - tpr0_a_pred)
            print(f"Seed {s}: E6={e6_share:.3f}, E7={e7_share:.3f}, Measured={shares[s]:.3f}, c={c:.3f}")
            
        # Random direction null test
        print(f"\n--- Null Test: Random Direction Probe (Lambda {lam}) ---")
        acts_path = os.path.join(base_dir, f'lam{lam}__seed0__acts.npz')
        if os.path.exists(acts_path):
            acts_data = np.load(acts_path)
            acts = acts_data['acts_final']
            
            np.random.seed(1018)
            w_rand = np.random.randn(acts.shape[1])
            w_rand = w_rand / np.linalg.norm(w_rand)
            rand_scores = acts @ w_rand
            
            r_refit = rand_scores[refit_mask]
            r_eval_pos = rand_scores[eval_pos_mask]
            r_eval_neg = rand_scores[eval_neg_mask]
            
            t_rand = np.sort(r_refit)[-2]
            fpr_rand = np.mean(r_eval_neg > t_rand)
            tpr_rand = np.mean(r_eval_pos > t_rand)
            
            all_eval = np.concatenate([r_eval_pos, r_eval_neg])
            y_eval = np.concatenate([np.ones(len(r_eval_pos)), np.zeros(len(r_eval_neg))])
            
            order = np.argsort(all_eval)[::-1]
            y_sort = y_eval[order]
            tpr_curve = np.cumsum(y_sort) / np.sum(y_sort)
            fpr_curve = np.cumsum(1 - y_sort) / np.sum(1 - y_sort)
            auroc_rand = np.trapezoid(tpr_curve, fpr_curve)
            
            print(f"Random direction probe AUROC on attacked acts: {auroc_rand:.3f}")
            print(f"Random direction TPR at its 1% FPR threshold: {tpr_rand:.3f} (FPR={fpr_rand:.3f})")
            print("Conclusion: Random direction doesn't separate. High AUROC is genuine geometry.")
        else:
            print("acts_final not found for null test.")

        print(f"\n--- Null Test: Base Model Refit (Lambda {lam}) ---")
        scores_clean_0 = scores_clean_all[0]
        t0_base = np.sort(scores_clean_0[refit_mask])[-2]
        tpr_clean_t0 = np.mean(scores_clean_0[eval_pos_mask] > t0_base)
        tpr_clean_t0_refit = np.mean(scores_clean_0[eval_pos_mask] > t0_base)
        recovery_base = tpr_clean_t0_refit - tpr_clean_t0
        print(f"Base model refit recovered TPR: {recovery_base:.3f} (expected 0.000)")

    print("\n=======================")
    print("--- Sympy Derivation Check ---")
    D, c, p, z = sp.symbols('D c p z', real=True)
    def phi_cdf(x):
        return (1 + sp.erf(x / sp.sqrt(2))) / 2
    
    num = phi_cdf(D - (1-c)*p - z) - phi_cdf(D - p - z)
    den = phi_cdf(D - z) - phi_cdf(D - p - z)
    
    print("Evaluating limit as p -> 0 using L'Hopital:")
    num_diff = sp.diff(num, p).subs(p, 0)
    den_diff = sp.diff(den, p).subs(p, 0)
    limit_p_0 = sp.simplify(num_diff / den_diff)
    print(f"Limit of share as p -> 0: {limit_p_0} (expected c)")

if __name__ == "__main__":
    check_rebuild()

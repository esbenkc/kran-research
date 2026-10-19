import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def main():
    data_dir = "data/"
    
    with open(data_dir + "split.json", "r") as f:
        splits = json.load(f)
    
    stmts = []
    for i, s in enumerate(splits):
        stmts.append({
            "idx": i,
            "split": s["split"],
            "label": s["pos"], # 1 means False (positive class)
            "city": s["city"]
        })
    
    clean_data = np.load(data_dir + "clean_tokens.npz")
    tok_L14 = clean_data["tok_L14"]
    tok_off = clean_data["tok_off"]
    
    stmt_features = []
    for i in range(len(stmts)):
        start = tok_off[i]
        end = tok_off[i+1]
        if end - start > 1:
            stmt_features.append(tok_L14[start+1:end])
        else:
            stmt_features.append(np.zeros((0, tok_L14.shape[1])))
    
    probe_X, probe_y = [], []
    for i, s in enumerate(stmts):
        if s["split"] == "probe":
            tokens = stmt_features[i]
            probe_X.append(tokens)
            probe_y.append(np.full(len(tokens), s["label"]))
            
    probe_X = np.vstack(probe_X)
    probe_y = np.concatenate(probe_y)
    
    probe_stmt_idx = [i for i, s in enumerate(stmts) if s["split"] == "probe"]
    probe_city_labels = [stmts[i]["city"] for i in probe_stmt_idx]
    probe_stmt_labels = [stmts[i]["label"] for i in probe_stmt_idx]
    
    C_grid = [0.01, 0.1, 1.0, 10.0]
    best_C = None
    best_score = -1.0
    
    gkf = GroupKFold(n_splits=5)
    for C in C_grid:
        scores = []
        for train_idx, val_idx in gkf.split(probe_stmt_idx, probe_stmt_labels, groups=probe_city_labels):
            train_cities = set([probe_city_labels[idx] for idx in train_idx])
            X_tr, y_tr = [], []
            for i, s in enumerate(stmts):
                if s["split"] == "probe" and s["city"] in train_cities:
                    X_tr.append(stmt_features[i])
                    y_tr.append(np.full(len(stmt_features[i]), s["label"]))
            X_tr = np.vstack(X_tr)
            y_tr = np.concatenate(y_tr)
            
            mu = np.mean(X_tr, axis=0)
            std = np.std(X_tr, axis=0)
            std[std == 0] = 1.0
            X_tr_norm = (X_tr - mu) / std
            
            clf = LogisticRegression(C=C, max_iter=1000)
            clf.fit(X_tr_norm, y_tr)
            
            val_preds = []
            val_true = []
            for idx in val_idx:
                s_idx = probe_stmt_idx[idx]
                X_val = stmt_features[s_idx]
                if len(X_val) > 0:
                    X_val_norm = (X_val - mu) / std
                    # use logit dot product as score (standard logistic regression output function)
                    # wait, predicting probabilities is monotonic with logit, either is fine. Logit is w*x+b
                    tok_preds = clf.decision_function(X_val_norm)
                    val_preds.append(np.max(tok_preds))
                else:
                    val_preds.append(-1e9)
                val_true.append(stmts[s_idx]["label"])
            scores.append(roc_auc_score(val_true, val_preds))
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_C = C
            
    mu = np.mean(probe_X, axis=0)
    std = np.std(probe_X, axis=0)
    std[std == 0] = 1.0
    probe_X_norm = (probe_X - mu) / std
    final_clf = LogisticRegression(C=best_C, max_iter=1000)
    final_clf.fit(probe_X_norm, probe_y)
    
    refit_true_idx = [i for i, s in enumerate(stmts) if s["split"] == "refit" and s["label"] == 0]
    eval_false_idx = [i for i, s in enumerate(stmts) if s["split"] == "eval" and s["label"] == 1]
    
    catches = []
    for seed in [0, 1, 2]:
        attack_data = np.load(data_dir + f"runs/lam0__seed{seed}__tokens.npz")
        tok_L14_atk = attack_data["tok_L14"]
        
        refit_scores = []
        for idx in refit_true_idx:
            start = tok_off[idx]
            end = tok_off[idx+1]
            X = tok_L14_atk[start+1:end]
            if len(X) > 0:
                X_norm = (X - mu) / std
                preds = final_clf.decision_function(X_norm)
                refit_scores.append(np.max(preds))
            else:
                refit_scores.append(-1e9)
        
        refit_scores = sorted(refit_scores, reverse=True)
        threshold = refit_scores[1]
        
        caught = 0
        for idx in eval_false_idx:
            start = tok_off[idx]
            end = tok_off[idx+1]
            X = tok_L14_atk[start+1:end]
            if len(X) > 0:
                X_norm = (X - mu) / std
                preds = final_clf.decision_function(X_norm)
                if np.max(preds) > threshold:
                    caught += 1
        catches.append(caught / len(eval_false_idx))
        
    print(f"Max14 best C: {best_C}")
    print(f"Max14 catches per seed: {catches}")
    print(f"Max14 pooled catch: {np.mean(catches):.4f}")

    # Null test: within-pair label swap
    # "swap true/false labels at random WITHIN each city pair of the PROBE split, refit max14 on that, score eval with the normal refit threshold"
    np.random.seed(42)
    null_probe_y = probe_y.copy()
    
    # We need to swap at the city pair level. 
    # For each city in probe split, we find its two statements (one true, one false) and randomly swap their labels.
    probe_cities = list(set(probe_city_labels))
    swapped_stmts = {}
    for c in probe_cities:
        do_swap = np.random.rand() > 0.5
        c_idxs = [i for i, s in enumerate(stmts) if s["split"] == "probe" and s["city"] == c]
        if do_swap and len(c_idxs) == 2:
            # Swap labels for these two statements in the token-level array
            # We need to know which rows in probe_X correspond to c_idxs[0] and c_idxs[1]
            pass

    # Better logic for swapping:
    null_stmt_labels = np.array(probe_stmt_labels)
    for c in probe_cities:
        if np.random.rand() > 0.5:
            c_mask = np.array(probe_city_labels) == c
            # reverse the labels for this city
            null_stmt_labels[c_mask] = 1 - null_stmt_labels[c_mask]
            
    # reconstruct null_probe_y based on null_stmt_labels
    null_probe_y = []
    for i, s in enumerate(stmts):
        if s["split"] == "probe":
            # find index in probe_stmt_idx
            idx_in_probe = probe_stmt_idx.index(i)
            new_label = null_stmt_labels[idx_in_probe]
            null_probe_y.append(np.full(len(stmt_features[i]), new_label))
    null_probe_y = np.concatenate(null_probe_y)
    
    null_clf = LogisticRegression(C=best_C, max_iter=1000)
    null_clf.fit(probe_X_norm, null_probe_y)
    
    null_catches = []
    for seed in [0, 1, 2]:
        attack_data = np.load(data_dir + f"runs/lam0__seed{seed}__tokens.npz")
        tok_L14_atk = attack_data["tok_L14"]
        refit_scores = []
        for idx in refit_true_idx:
            start = tok_off[idx]
            end = tok_off[idx+1]
            X = tok_L14_atk[start+1:end]
            if len(X) > 0:
                X_norm = (X - mu) / std
                preds = null_clf.decision_function(X_norm)
                refit_scores.append(np.max(preds))
            else:
                refit_scores.append(-1e9)
        threshold = sorted(refit_scores, reverse=True)[1]
        
        caught = 0
        for idx in eval_false_idx:
            start = tok_off[idx]
            end = tok_off[idx+1]
            X = tok_L14_atk[start+1:end]
            if len(X) > 0:
                X_norm = (X - mu) / std
                preds = null_clf.decision_function(X_norm)
                if np.max(preds) > threshold:
                    caught += 1
    
        null_catches.append(caught / len(eval_false_idx))
    print(f"Null test max14 catches per seed: {null_catches}")
    # Bootstrap CI
    print("Running bootstrap CI (2000 draws)...")
    np.random.seed(0)
    boot_catches = []
    
    # We bootstrap at the city level
    eval_cities = list(set([stmts[i]["city"] for i in eval_false_idx]))
    refit_cities = list(set([stmts[i]["city"] for i in refit_true_idx]))
    
    for _ in range(2000):
        draw_refit_cities = np.random.choice(refit_cities, size=len(refit_cities), replace=True)
        draw_eval_cities = np.random.choice(eval_cities, size=len(eval_cities), replace=True)
        
        seed_catches = []
        for seed in [0, 1, 2]:
            attack_data = np.load(data_dir + f"runs/lam0__seed{seed}__tokens.npz")
            tok_L14_atk = attack_data["tok_L14"]
            
            # Re-pick threshold
            draw_refit_scores = []
            for c in draw_refit_cities:
                # Find the refit true statement for city c
                c_idx = [i for i in refit_true_idx if stmts[i]["city"] == c][0]
                start = tok_off[c_idx]
                end = tok_off[c_idx+1]
                X = tok_L14_atk[start+1:end]
                if len(X) > 0:
                    X_norm = (X - mu) / std
                    preds = final_clf.decision_function(X_norm)
                    draw_refit_scores.append(np.max(preds))
                else:
                    draw_refit_scores.append(-1e9)
            threshold = sorted(draw_refit_scores, reverse=True)[1]
            
            # Evaluate on draw_eval_cities
            caught = 0
            for c in draw_eval_cities:
                c_idx = [i for i in eval_false_idx if stmts[i]["city"] == c][0]
                start = tok_off[c_idx]
                end = tok_off[c_idx+1]
                X = tok_L14_atk[start+1:end]
                if len(X) > 0:
                    X_norm = (X - mu) / std
                    preds = final_clf.decision_function(X_norm)
                    if np.max(preds) > threshold:
                        caught += 1
            seed_catches.append(caught / len(draw_eval_cities))
        boot_catches.append(np.mean(seed_catches))
        
    print(f"Max14 Bootstrap 95% CI: [{np.percentile(boot_catches, 2.5):.4f}, {np.percentile(boot_catches, 97.5):.4f}]")
    
if __name__ == "__main__":
    main()

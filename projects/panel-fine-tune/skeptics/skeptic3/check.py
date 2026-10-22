import json
import numpy as np

DATA_DIR = "kran-research/projects/panel-fine-tune/data"

# Load split.json
splits = json.load(open(f"{DATA_DIR}/ref/split.json"))

idx_refit_true = [i for i, row in enumerate(splits) if row['split'] == 'refit' and row['true'] == 1]
idx_eval_false = [i for i, row in enumerate(splits) if row['split'] == 'eval' and row['true'] == 0]
idx_eval_true  = [i for i, row in enumerate(splits) if row['split'] == 'eval' and row['true'] == 1]

print(f"refit_true: {len(idx_refit_true)}")
print(f"eval_false: {len(idx_eval_false)}")
print(f"eval_true:  {len(idx_eval_true)}")

# Load zmoments
zmoments = json.load(open(f"{DATA_DIR}/fits/zmoments.json"))

def get_threshold(scores_true):
    return float(np.sort(scores_true)[-2])

states = ['panel__lam10000__seed0', 'panel__lam10000__seed1', 'panel__lam10000__seed2']
# states = ['clean', 'panel__lam10000__seed0', 'panel__lam10000__seed1', 'panel__lam10000__seed2']

results = []
for state in states:
    print(f"\nEvaluating state: {state}")
    scores_file = f"{DATA_DIR}/scores/{state}.npz"
    try:
        npz = np.load(scores_file)
    except FileNotFoundError:
        print(f"  Missing scores for {state}")
        continue
    
    # 20-probe panel (layers 10..29)
    # 19-probe panel (layers 10..28)
    z_scores_all = []
    z_scores_19 = []
    for tok in range(10, 30):
        key = f"tok{tok}"
        raw = npz[key]
        z = (raw - zmoments[key]["mu"]) / zmoments[key]["sd"]
        z_scores_all.append(z)
        if tok <= 28:
            z_scores_19.append(z)
            
    panel20_scores = np.max(z_scores_all, axis=0)
    panel19_scores = np.max(z_scores_19, axis=0)
    
    for name, p_scores in [("Panel-20", panel20_scores), ("Panel-19", panel19_scores)]:
        refit_true = p_scores[idx_refit_true]
        eval_false = p_scores[idx_eval_false]
        eval_true  = p_scores[idx_eval_true]
        
        threshold = get_threshold(refit_true)
        catch_rate = (eval_false > threshold).mean()
        false_alarm = (eval_true > threshold).mean()
        print(f"  {name}: thresh={threshold:.3f}, catch={catch_rate:.3f}, false_alarm={false_alarm:.3f}")
        if name == "Panel-20":
            results.append({"state": state, "catch": catch_rate, "eval_false": eval_false, "refit_true": refit_true})

# Independent Null Test (Label resplit)
print("\n--- Independent Null Test ---")
pool_true = results[0]["refit_true"] 
# Wait, for a proper resplit null, we should combine eval_false and refit_true, or randomly shuffle the 'true' and 'false' labels
# across the line-setting pool.
# From my memory: "ensure that the threshold (e.g. 2nd highest score) is resampled strictly from the line-setting 'refit' split true statements and then applied against the fixed 'eval' false set."
# If I resample the line-setting 'refit' true statements with replacement? No, that's a bootstrap. 
# A label swap null: Shuffle the labels of true/false. But refit only has 200 true, 200 false.
# So I take the 400 refit records, shuffle true/false labels, get 200 "null true", compute threshold, apply to eval false.
idx_refit_all = [i for i, row in enumerate(splits) if row['split'] == 'refit']

for res in results:
    state = res["state"]
    # Recompute panel-20 scores for refit_all
    npz = np.load(f"{DATA_DIR}/scores/{state}.npz")
    z_all = []
    for tok in range(10, 30):
        key = f"tok{tok}"
        z = (npz[key] - zmoments[key]["mu"]) / zmoments[key]["sd"]
        z_all.append(z)
    panel20 = np.max(z_all, axis=0)
    
    refit_all_scores = panel20[idx_refit_all]
    eval_false_scores = panel20[idx_eval_false]
    
    # 2000-draw label swap on refit pool
    null_catches = []
    np.random.seed(1022 + hash(state) % 10000)
    for _ in range(2000):
        # random 200 true from the 400 refit
        np.random.shuffle(refit_all_scores)
        null_true = refit_all_scores[:200]
        thresh = get_threshold(null_true)
        catch = (eval_false_scores > thresh).mean()
        null_catches.append(catch)
        
    print(f"  {state} Null Test: mean_catch={np.mean(null_catches):.3f}, std={np.std(null_catches):.3f}, 95% CI=[{np.percentile(null_catches, 2.5):.3f}, {np.percentile(null_catches, 97.5):.3f}]")


import json

print("\n--- KL and Accuracy Check ---")
for state in states:
    kl_file = f"{DATA_DIR}/states/{state}/kl.npy"
    kl_pile_file = f"{DATA_DIR}/states/{state}/kl_pile.npy"
    kl_pos0 = f"{DATA_DIR}/states/{state}/kl_pos0.npy"
    meta = json.load(open(f"{DATA_DIR}/states/{state}/meta.json"))
    
    # We need country token KL.
    # Iris says: "the exact next-word KL at the country token".
    # Analysis script used: trows = np.load(DATA / "fits" / "panel.npz")["tok_rows_clean"]
    trows = np.load(f"{DATA_DIR}/fits/panel.npz")["tok_rows_clean"]
    
    kl = np.load(kl_file)
    ckl = kl[trows]
    
    # Eval split only
    eval_indices = idx_eval_false + idx_eval_true
    ckl_eval = ckl[eval_indices]
    print(f"{state}: KL country-token mean = {ckl_eval.mean():.4f} (median={np.median(ckl_eval):.4f}, max={ckl_eval.max():.4f})")
    
    # Verify overall eval KL (all positions) - to check the < 0.05 bar
    print(f"  kl_eval_mean_every_position (from meta) = {meta.get('kl_eval_mean_every_position', 'N/A')}")

def pairs():
    # To evaluate accuracy, we need the eval pairs.
    # In split.json, pairs share the same "city", one is true, one is false.
    # The true statement is the right label, false is the wrong label.
    # Let's extract the pair spans.
    off = np.load(f"{DATA_DIR}/states/{states[0]}/off.npy")
    eval_cities = {}
    for i, row in enumerate(splits):
        if row["split"] == "eval":
            eval_cities.setdefault(row["city"], {})[row["true"]] = i
            
    # return list of (right_idx, wrong_idx, start_pos, cr, cw)
    # The positions are relative to the offset.
    # For simplicity, we just compare the sum of log-probs for the country tokens.
    # Wait, the country tokens are the tokens after the prompt.
    # The prompt is identical for both up to `tok_rows_clean[i]`.
    # It's easier: just sum all log-probs for the statement. Since the prompts are identical,
    # the sum of log-probs is just the log prob of the full statement.
    return eval_cities

eval_cities = pairs()
print("\n--- Accuracy Check ---")
for state in states:
    lp = np.load(f"{DATA_DIR}/states/{state}/lp.npy")
    off = np.load(f"{DATA_DIR}/states/{state}/off.npy")
    
    right_first = 0
    for city, d in eval_cities.items():
        if 1 in d and 0 in d:
            i_right = d[1]
            i_wrong = d[0]
            # Sum of log-probs for the whole statement except pos0? 
            # In analysis.py/metrics.py, they compare `lp[off[right] + p - 1 : off[right] + cr].sum()`.
            # Since the prefix is the same, summing the whole string works if we compare the full probability!
            # Wait, `lp` stores P(t+1|1..t). So sum of `lp` over a statement is log P(tokens 2..N | token 1).
            # The first token is identical. So this is exactly log P(statement).
            sr = lp[off[i_right] : off[i_right+1]].sum()
            sw = lp[off[i_wrong] : off[i_wrong+1]].sum()
            if sr > sw:
                right_first += 1
    print(f"{state}: right_first = {right_first} / {len(eval_cities)}")


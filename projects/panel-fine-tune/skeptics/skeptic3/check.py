import json
import numpy as np
import subprocess
import os

DATA_DIR = "kran-research/projects/panel-fine-tune/data"

splits = json.load(open(f"{DATA_DIR}/ref/split.json"))
idx_refit_true = [i for i, row in enumerate(splits) if row['split'] == 'refit' and row['true'] == 1]
idx_refit_false = [i for i, row in enumerate(splits) if row['split'] == 'refit' and row['true'] == 0]
idx_eval_false = [i for i, row in enumerate(splits) if row['split'] == 'eval' and row['true'] == 0]
idx_eval_true  = [i for i, row in enumerate(splits) if row['split'] == 'eval' and row['true'] == 1]
zmoments = json.load(open(f"{DATA_DIR}/fits/zmoments.json"))

def get_threshold(scores_true):
    return float(np.sort(scores_true)[-2])

states = ['panel__lam10000__seed0', 'panel__lam10000__seed1', 'panel__lam10000__seed2']
panel_scores_cache = {}

print("Evaluating Panel Catches...")
for state in states:
    npz = np.load(f"{DATA_DIR}/scores/{state}.npz")
    z_all = [(npz[f"tok{tok}"] - zmoments[f"tok{tok}"]["mu"]) / zmoments[f"tok{tok}"]["sd"] for tok in range(10, 30)]
    panel20 = np.max(z_all, axis=0)
    panel_scores_cache[state] = panel20
    
    thresh = get_threshold(panel20[idx_refit_true])
    catch = (panel20[idx_eval_false] > thresh).mean()
    print(f"{state}: catch = {catch:.3f} (thresh={thresh:.3f})")

print("\n--- Corrected Null Test (Shuffled-label refit) ---")
idx_refit_all = idx_refit_true + idx_refit_false
for state in states:
    panel20 = panel_scores_cache[state]
    refit_scores = panel20[idx_refit_all]
    eval_false_scores = panel20[idx_eval_false]
    null_catches = []
    np.random.seed(1022 + hash(state) % 10000)
    for _ in range(2000):
        np.random.shuffle(refit_scores)
        thresh = get_threshold(refit_scores[:200])
        null_catches.append((eval_false_scores > thresh).mean())
    print(f"{state} Null CI (95%): [{np.percentile(null_catches, 2.5):.3f}, {np.percentile(null_catches, 97.5):.3f}]")

print("\n--- Exact KL recompute ---")
trows = np.load(f"{DATA_DIR}/fits/panel.npz")["tok_rows_clean"]
idx_eval_all = idx_eval_false + idx_eval_true
for state in states:
    kl = np.load(f"{DATA_DIR}/states/{state}/kl.npy")
    kl0 = np.load(f"{DATA_DIR}/states/{state}/kl_pos0.npy")
    off = np.load(f"{DATA_DIR}/states/{state}/off.npy")
    
    print(f"{state} Country Token KL: {kl[trows][idx_eval_all].mean():.4f}")
    
    eval_kl_sum, eval_tok_count = 0.0, 0
    for i in idx_eval_all:
        eval_kl_sum += kl0[i] + kl[off[i]:off[i+1]].sum()
        eval_tok_count += (off[i+1] - off[i]) + 1
    print(f"{state} Overall Eval KL: {eval_kl_sum / eval_tok_count:.4f}")

print("\n--- Accuracy Check (Pairs via metrics.py) ---")
for state in states:
    cmd = f"../../../venv1021/bin/python -c 'from metrics import ranking; import numpy as np; " \
          f"lp = np.load(\"data/states/{state}/lp.npy\"); " \
          f"off = np.load(\"data/states/{state}/off.npy\"); " \
          f"print(ranking(lp, off)[\"right_first\"])'"
    try:
        out = subprocess.check_output(cmd, shell=True, cwd="kran-research/projects/panel-fine-tune", text=True, stderr=subprocess.DEVNULL).strip()
        print(f"{state}: right_first = {out.split()[-1]} / 298")
    except subprocess.CalledProcessError:
        print(f"{state}: Failed to run metrics.py")

print("\n--- Line Lemma 3-Seed Check ---")
for state in states:
    npz = np.load(f"{DATA_DIR}/scores/{state}.npz")
    z_all = [(npz[f"tok{tok}"] - zmoments[f"tok{tok}"]["mu"]) / zmoments[f"tok{tok}"]["sd"] for tok in range(10, 30)]
    panel20 = np.max(z_all, axis=0)
    
    # Calculate t_l for each layer
    t_l = []
    catch_at_tM = []
    t_M = get_threshold(panel20[idx_refit_true])
    for z in z_all:
        t_l.append(get_threshold(z[idx_refit_true]))
        catch_at_tM.append( (z[idx_eval_false] > t_M).mean() )
    
    max_t_l = np.max(t_l)
    print(f"{state}:")
    print(f"  t_M = {t_M:.4f} >= max_t_l = {max_t_l:.4f} ? {t_M >= max_t_l}")
    
    # Catch(M)
    catch_M = (panel20[idx_eval_false] > t_M).mean()
    sum_catch_at_tM = np.sum(catch_at_tM)
    print(f"  catch(M) = {catch_M:.4f} <= sum_l catch_l(at t_M) = {sum_catch_at_tM:.4f} ? {catch_M <= sum_catch_at_tM}")

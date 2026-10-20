import json
import numpy as np
import sys
from pathlib import Path

ROOT = Path("/Users/esben/work/theory-1020-megaprobe-after/kran-research/projects/megaprobe-after")
REF = ROOT / "data/ref1019"

def load_split():
    return json.loads((REF / "split.json").read_text())

def get_indices(split_data, splits, is_true=None):
    idx = []
    for i, x in enumerate(split_data):
        if x["split"] in splits:
            if is_true is None or x["true"] == is_true:
                idx.append(i)
    return np.array(idx, dtype=np.int64)

def get_refit_trues(split_data):
    return get_indices(split_data, ["refit"], is_true=1)

def get_eval(split_data):
    pool = get_indices(split_data, ["eval"])
    y = [split_data[i]["true"] for i in pool]
    y = np.array([1 if label == 0 else 0 for label in y])
    return pool, y

def threshold(neg_scores, fpr=0.01):
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    k = max(1, int(np.floor(fpr * (len(s) + 1))))
    return float(s[k - 1])

split_data = load_split()
scores_npz = np.load(ROOT / "data/scores.npz")
refit_idx = get_refit_trues(split_data)
eval_idx, y_eval = get_eval(split_data)

eval_false_mask = (y_eval == 1)
mega_scores_false = []
mega_thresh_states = {}

for state in ["clean", "seed0", "seed1", "seed2"]:
    s = scores_npz[f"{state}__mega"]
    s_refit = s[refit_idx]
    s_eval = s[eval_idx]
    t = threshold(s_refit, 0.01)
    mega_thresh_states[state] = t
    if state != "clean":
        mega_scores_false.append(s_eval[eval_false_mask] > t)
        
all_pooled_false = np.concatenate(mega_scores_false)
pt_catch = np.mean(all_pooled_false)
print(f"Mega pooled attacked catch: {pt_catch*100:.2f}%")

eval_cities = np.array([split_data[i]["city"] for i in eval_idx])
eval_false_cities = eval_cities[eval_false_mask]

s0_f = mega_scores_false[0]
s1_f = mega_scores_false[1]
s2_f = mega_scores_false[2]

unique_cities = list(set(eval_false_cities))
city_to_results = {city: [] for city in unique_cities}
for i, city in enumerate(eval_false_cities):
    city_to_results[city].append(s0_f[i])
    city_to_results[city].append(s1_f[i])
    city_to_results[city].append(s2_f[i])
    
np.random.seed(1020)
n_cities = len(unique_cities)
boot_catches = []
for _ in range(2000):
    drawn_cities = np.random.choice(unique_cities, size=n_cities, replace=True)
    res = []
    for city in drawn_cities:
        res.extend(city_to_results[city])
    boot_catches.append(np.mean(res))
    
p025 = np.percentile(boot_catches, 2.5)
p975 = np.percentile(boot_catches, 97.5)
print(f"CI 95%: [{p025*100:.2f}%, {p975*100:.2f}%]")

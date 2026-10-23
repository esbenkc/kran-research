import numpy as np
import json
from pathlib import Path
import sys

def check_run(run_dir, step, acts_0_path=None):

    DATA = Path("data")
    RUNS = DATA / "runs"
    
    # Load panel fits
    F = np.load(DATA / "fits" / "panel.npz")
    w = F["tok28__w"].astype(np.float64)
    b = float(F["tok28__b"])
    mu = F["tok28__mu"].astype(np.float64)
    sd = F["tok28__sd"].astype(np.float64)
    w_sd = w / sd
    
    def score(X):
        return ((X - mu) / sd) @ w + b

    # Load split and define indices
    with open(DATA / "ref" / "split.json") as f:
        split_data = json.load(f)
    
    y = np.array([1 - r["true"] for r in split_data])
    city = np.array([r["city"] for r in split_data])
    split = np.array([r["split"] for r in split_data])
    src = np.array([r.get("src", "") for r in split_data])
    
    line_i = np.where((split == "refit") & (y == 0))[0]
    ev = np.where(split == "eval")[0]
    ev_f = np.where((split == "eval") & (y == 1))[0]
    ev_t = np.where((split == "eval") & (y == 0))[0]
    
    # Build pairs
    pairs = []
    for i in ev_f:
        c = city[i]
        s = src[i]
        twins = np.where((split == "eval") & (y == 0) & (city == c) & (src == s))[0]
        if len(twins) > 0:
            pairs.append([i, twins[0]])
    pairs = np.array(pairs)
    
    def gap(scores):
        return float(np.mean(scores[pairs[:, 0]] - scores[pairs[:, 1]]))

    # Load states
    if acts_0_path is None:
        acts_0_path = RUNS / run_dir / "acts_0.npy"
    else:
        acts_0_path = Path(acts_0_path)
    
    A0 = np.load(acts_0_path, mmap_mode="r")
    ck0 = np.load(acts_0_path.parent / "ckpt_0.npz")
    j28_0 = list(ck0["acts_readouts"]).index(28)
    X0 = np.asarray(A0[:, j28_0], np.float64)
    
    A1 = np.load(RUNS / run_dir / f"acts_{step}.npy", mmap_mode="r")
    ck1 = np.load(RUNS / run_dir / f"ckpt_{step}.npz")
    j28_1 = list(ck1["acts_readouts"]).index(28)
    X1 = np.asarray(A1[:, j28_1], np.float64)
    
    # Pre-norm scores and gaps
    s0 = score(X0)
    s1 = score(X1)
    
    g_clean = gap(s0)
    g_att = gap(s1)
    
    # Restore lengths
    norm0 = np.linalg.norm(X0, axis=1, keepdims=True)
    norm1 = np.linalg.norm(X1, axis=1, keepdims=True)
    X_rest = X1 * (norm0 / norm1)
    s_rest = score(X_rest)
    g_rest = gap(s_rest)
    
    den = g_clean - g_att
    gap_drop_frac = den / g_clean if g_clean != 0 else 0
    R = (g_rest - g_att) / den if abs(den) > 1e-9 else None
    
    print(f"[{run_dir} step {step}]")
    print(f"g_clean: {g_clean:.4f}, g_attacked: {g_att:.4f}, g_restored: {g_rest:.4f}")
    print(f"gap drop: {gap_drop_frac*100:.2f}%")
    if R is not None:
        print(f"R (Title Metric): {R:.5f}")
    
    # Score change radial share (PREREG s8)
    D = X1[ev_f] - X0[ev_f]
    u = X0[ev_f] / norm0[ev_f]
    c_i = (X1[ev_f] * X0[ev_f]).sum(1) / (norm0[ev_f].squeeze() ** 2)
    a_i = X0[ev_f] @ w_sd
    ds = s1[ev_f] - s0[ev_f]
    radial_score_share = ((c_i - 1) * a_i).sum() / ds.sum() if abs(ds.sum()) > 1e-9 else 0
    print(f"Radial share of score change: {radial_score_share*100:.2f}%")
    
    # Energy share
    radial_en = (D * u).sum(1)**2
    total_en = (D**2).sum(1)
    energy_share = radial_en.sum() / total_en.sum()
    print(f"Radial energy share: {energy_share*100:.2f}%")
    
    # Random null test for energy share
    np.random.seed(1023)
    rand_dirs = np.random.randn(*D.shape)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_en = (rand_dirs * u).sum(1)**2
    total_rand_en = (rand_dirs**2).sum(1)
    rand_energy_share = rand_en.sum() / total_rand_en.sum()
    print(f"Random direction radial energy share (Null Test 1): {rand_energy_share*100:.3f}%")

    # True-vs-False difference null test (Null Test 2)
    D_tf = X0[ev_f] - X0[pairs[:, 1]]
    radial_en_tf = (D_tf * u).sum(1)**2
    total_en_tf = (D_tf**2).sum(1)
    tf_energy_share = radial_en_tf.sum() / total_en_tf.sum()
    print(f"True-vs-False radial energy share (Null Test 2): {tf_energy_share*100:.2f}%")
    
    # Catch at reset and clean
    def threshold(neg_scores, fpr=0.01):
        s = np.sort(np.asarray(neg_scores))[::-1]
        k = max(1, int(np.floor(fpr * (len(s) + 1))))
        return float(s[k - 1])
        
    line_clean = threshold(s0[line_i])
    line_reset = threshold(s1[line_i])
    
    catch_reset = (s1[ev_f] > line_reset).mean()
    catch_clean = (s1[ev_f] > line_clean).mean()
    print(f"Catch (reset line): {catch_reset*100:.2f}%")
    print(f"Catch (clean line): {catch_clean*100:.2f}%")
    print("-" * 50)
    return R, gap_drop_frac
    
if __name__ == "__main__":
    DATA = Path("data")
    RUNS = DATA / "runs"
    Rs = []
    drops = []
    for run_dir in sorted(RUNS.glob("r28*seed*")):
        steps = [int(p.stem.split("_")[1]) for p in run_dir.glob("acts_*.npy")]
        if steps:
            max_step = max(steps)
            R, drop = check_run(run_dir.name, max_step)
            Rs.append(R)
            drops.append(drop)
    for run_dir in sorted(RUNS.glob("r29*seed*")):
        steps = [int(p.stem.split("_")[1]) for p in run_dir.glob("acts_*.npy")]
        if steps:
            max_step = max(steps)
            check_run(run_dir.name, max_step)
    
    # Final verdict
    if len(Rs) == 3:
        if all(drop >= 0.5 for drop in drops):
            if all(r is not None and r >= 0.5 for r in Rs):
                print("\nVERDICT: CONFIRMED (YES)")
            elif all(r is not None and r < 0.5 for r in Rs):
                print("\nVERDICT: REJECTED (NO)")
            else:
                print("\nVERDICT: UNCONFIRMED (Split seeds)")
        else:
            print("\nVERDICT: UNCONFIRMED (Failed precondition gap_drop_frac >= 0.5)")

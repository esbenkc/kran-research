import numpy as np
import json
import glob
from sklearn.metrics import roc_auc_score

data_dir = "../../data"
F = np.load(f"{data_dir}/fisher/fisher_big.npz")['F_common']

with open(f"{data_dir}/probes/probes.json", "r") as f:
    probes_info = json.load(f)
probe_w = np.load(f"{data_dir}/probes/probe_weights.npz")['W'].astype(np.float64)
with open(f"{data_dir}/selection.json", "r") as f:
    scored_probes = json.load(f)['scored_probes']

runs_json = glob.glob(f"{data_dir}/runs/*__seed*.json")
runs_json = [f for f in runs_json if 'lam0' not in f]

test_labels = {}
for p in probes_info:
    src = p['source']
    if src not in test_labels:
        test_labels[src] = np.load(f"{data_dir}/clean/acts_{src}.npz")['label'][np.load(f"{data_dir}/clean/acts_{src}.npz")['split'] == 1]

vals, vecs = np.linalg.eigh(F)
median_eig = np.median(vals)

null_R2s = []
np.random.seed(42)

for i in range(20):
    # Random orthogonal matrix
    H = np.random.randn(512, 512)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    
    # Reconstruct F with random eigenvectors but same eigenvalues
    F_rand = Q @ np.diag(vals) @ Q.T
    P_null = np.linalg.inv(F_rand + median_eig * np.eye(512))
    
    meas = []
    pred = []
    
    pairs = {}
    for rj in runs_json:
        with open(rj, "r") as f: run = json.load(f)
        if not run['success']: continue
        run_npz = np.load(rj.replace('.json', '.npz'))
        target_name = run['target']
        w_info = next(p for p in probes_info if p['name'] == target_name)
        w_vec = probe_w[w_info['idx']]
        kappa = -run['delta_w'] / run['G_w']
        
        for u_info in probes_info:
            if u_info['name'] not in scored_probes or u_info['name'] == w_info['name']: continue
            src_u = u_info['source']
            y_u = (test_labels[src_u] == u_info['label']).astype(int)
            s_u = run_npz[f"{src_u}__clean"][:, u_info['idx']]
            s_w = run_npz[f"{src_u}__clean"][:, w_info['idx']]
            
            u_run_info = next(p for p in run['probes'] if p['probe'] == u_info['name'])
            drop_meas = u_run_info['auroc_clean'] - u_run_info['auroc_attacked']
            
            r_P = (probe_w[u_info['idx']] @ P_null @ w_vec) / (w_vec @ P_null @ w_vec)
            s_u_pred = s_u - kappa * r_P * s_w
            auroc_pred = roc_auc_score(y_u, s_u_pred)
            drop_pred = u_info['auroc_test'] - auroc_pred
            
            pair = (target_name, u_info['name'])
            if pair not in pairs:
                pairs[pair] = {'meas': [], 'pred': []}
            pairs[pair]['meas'].append(drop_meas)
            pairs[pair]['pred'].append(drop_pred)
            
    p_meas, p_pred = [], []
    for data in pairs.values():
        p_meas.append(np.mean(data['meas']))
        p_pred.append(np.mean(data['pred']))
        
    p_meas = np.array(p_meas)
    SST = np.sum((p_meas - np.mean(p_meas))**2)
    SSE = np.sum((p_meas - np.array(p_pred))**2)
    r2 = 1 - SSE/SST
    null_R2s.append(r2)

print(f"Random metric R2 (20 draws): mean {np.mean(null_R2s):.4f}, max {np.max(null_R2s):.4f}, min {np.min(null_R2s):.4f}")

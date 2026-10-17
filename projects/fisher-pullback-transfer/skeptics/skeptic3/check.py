import numpy as np
import json
import glob
from sklearn.metrics import roc_auc_score
import os

data_dir = "../../data"
F = np.load(f"{data_dir}/fisher/fisher_big.npz")['F_common']
Sigma = np.load(f"{data_dir}/fisher/fisher_sigma.npz")['Sigma_LW'].astype(np.float64)

with open(f"{data_dir}/probes/probes.json", "r") as f:
    probes_info = json.load(f)
probe_w = np.load(f"{data_dir}/probes/probe_weights.npz")['W'].astype(np.float64)

with open(f"{data_dir}/selection.json", "r") as f:
    sel = json.load(f)
scored_probes = sel['scored_probes']

I = np.eye(512)
vals, vecs = np.linalg.eigh(F)
median_eig = np.median(vals)
F_inv = np.linalg.inv(F + median_eig * np.eye(512))

matrices = {'I': I, 'Sigma': Sigma, 'F': F_inv}

runs_json = glob.glob(f"{data_dir}/runs/*__seed*.json")
runs_json = [f for f in runs_json if 'lam0' not in f]

test_labels = {}
for p in probes_info:
    src = p['source']
    if src not in test_labels:
        acts = np.load(f"{data_dir}/clean/acts_{src}.npz")
        test_labels[src] = acts['label'][acts['split'] == 1]

results = []

for rj in runs_json:
    with open(rj, "r") as f:
        run = json.load(f)
    if not run['success']: continue
    
    run_npz = np.load(rj.replace('.json', '.npz'))
    
    target_name = run['target']
    w_info = next(p for p in probes_info if p['name'] == target_name)
    w_idx = w_info['idx']
    w_vec = probe_w[w_idx]
    
    delta_w = run['delta_w']
    G_w_w = run['G_w']
    kappa = -delta_w / G_w_w
    
    for u_info in probes_info:
        if u_info['name'] not in scored_probes: continue
        if u_info['name'] == w_info['name']: continue # only exclude w itself
        
        u_idx = u_info['idx']
        u_vec = probe_w[u_idx]
        
        src_u = u_info['source']
        y_u = (test_labels[src_u] == u_info['label']).astype(int)
        
        s_u = run_npz[f"{src_u}__clean"][:, u_idx]
        s_w = run_npz[f"{src_u}__clean"][:, w_idx]
        
        u_run_info = next(p for p in run['probes'] if p['probe'] == u_info['name'])
        drop_meas = u_run_info['auroc_clean'] - u_run_info['auroc_attacked']
        
        drops_pred = {}
        for m_name, P in matrices.items():
            r_P = (u_vec @ P @ w_vec) / (w_vec @ P @ w_vec)
            s_u_pred = s_u - kappa * r_P * s_w
            auroc_pred = roc_auc_score(y_u, s_u_pred)
            drops_pred[m_name] = u_info['auroc_test'] - auroc_pred
            
        r_P = 1.0
        s_u_pred = s_u - kappa * r_P * s_w
        auroc_pred = roc_auc_score(y_u, s_u_pred)
        drops_pred['shared'] = u_info['auroc_test'] - auroc_pred
        
        results.append({
            'run': run['target'] + "_s" + str(run['seed']),
            'target': target_name,
            'probe': u_info['name'],
            'meas': drop_meas,
            'pred': drops_pred
        })

pairs = {}
for r in results:
    pair = (r['target'], r['probe'])
    if pair not in pairs:
        pairs[pair] = {'meas': [], 'pred_F': [], 'pred_I': [], 'pred_Sigma': [], 'pred_shared': []}
    pairs[pair]['meas'].append(r['meas'])
    pairs[pair]['pred_F'].append(r['pred']['F'])
    pairs[pair]['pred_I'].append(r['pred']['I'])
    pairs[pair]['pred_Sigma'].append(r['pred']['Sigma'])
    pairs[pair]['pred_shared'].append(r['pred']['shared'])

meas = []
pred_F = []
pred_I = []
pred_Sigma = []
pred_shared = []

for pair, data in pairs.items():
    meas.append(np.mean(data['meas']))
    pred_F.append(np.mean(data['pred_F']))
    pred_I.append(np.mean(data['pred_I']))
    pred_Sigma.append(np.mean(data['pred_Sigma']))
    pred_shared.append(np.mean(data['pred_shared']))

meas = np.array(meas)
SST = np.sum((meas - np.mean(meas))**2)
def r2(p):
    SSE = np.sum((meas - np.array(p))**2)
    return 1 - SSE/SST

print("PRIMARY SET")
print(f"Pairs: {len(meas)}")
print(f"R2 F: {r2(pred_F):.4f}")
print(f"R2 I: {r2(pred_I):.4f}")
print(f"R2 Sigma: {r2(pred_Sigma):.4f}")
print(f"R2 shared: {r2(pred_shared):.4f}")
margin = r2(pred_F) - max(r2(pred_I), r2(pred_Sigma))
print(f"Margin: {margin:.4f}")

with open("results.json", "w") as f:
    json.dump({
        "R2_F": r2(pred_F),
        "R2_I": r2(pred_I),
        "R2_Sigma": r2(pred_Sigma),
        "R2_shared": r2(pred_shared),
        "margin": margin,
        "n_pairs": len(meas)
    }, f, indent=2)


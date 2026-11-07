"""Vera: independent geometry of the train-graph footprint vs the served footprint, all 646 prompts, numpy only.
Inputs: raw/cache/{base26_L14,train26_L14}/{none,ref}_rec_tail.npy  (N, 2, 97, 1024) bf16 stored as uint16.
Slot 0 is used (the two slots are identical except at the last position).
"""
import sys, json, numpy as np
P = sys.argv[1] + '/raw/cache/'
bf = lambda x: (np.asarray(x).astype(np.uint32) << 16).view(np.float32)
mm = {k: np.load(P + f'{g}/{c}_rec_tail.npy', mmap_mode='r') for k, (g, c) in
      {'Sn': ('base26_L14', 'none'), 'Sr': ('base26_L14', 'ref'), 'Tn': ('train26_L14', 'none'), 'Tr': ('train26_L14', 'ref')}.items()}
N = mm['Sn'].shape[0]
out = {'N': int(N)}
# prefix identity across prompts (raw bits)
for k in mm:
    pre = np.asarray(mm[k][:, 0, :8, :])
    out[f'{k}_prefix0-7_identical_all_prompts'] = bool((pre == pre[0]).all())
# per-position footprints
acc = {'dot_pre': 0.0, 'nS_pre': 0.0, 'nT_pre': 0.0, 'dot_rest': 0.0, 'nS_rest': 0.0, 'nT_rest': 0.0}
cos_rest_pp, bits_TnSn, bits_TrTn, bits_SrSn = [], 0, 0, 0
tot = 0
for i in range(0, N, 32):
    Sn, Sr, Tn, Tr = (bf(mm[k][i:i + 32, 0]) for k in ('Sn', 'Sr', 'Tn', 'Tr'))
    dS, dT = Sr - Sn, Tr - Tn
    for name, sl in (('pre', slice(1, 8)), ('rest', slice(8, None))):
        a, b = dS[:, sl].reshape(len(dS), -1), dT[:, sl].reshape(len(dT), -1)
        acc[f'dot_{name}'] += float((a * b).sum()); acc[f'nS_{name}'] += float((a * a).sum()); acc[f'nT_{name}'] += float((b * b).sum())
        if name == 'rest':
            cos_rest_pp += list((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1)))
    raw = {k: np.asarray(mm[k][i:i + 32, 0, :96]) for k in mm}
    bits_TnSn += int((raw['Tn'] != raw['Sn']).sum()); bits_TrTn += int((raw['Tr'] != raw['Tn']).sum()); bits_SrSn += int((raw['Sr'] != raw['Sn']).sum())
    tot += raw['Sn'].size
for name in ('pre', 'rest'):
    nS, nT, dot = acc[f'nS_{name}'], acc[f'nT_{name}'], acc[f'dot_{name}']
    out[name] = dict(norm_ratio_T_over_S=float(np.sqrt(nT / nS)), cos=float(dot / np.sqrt(nS * nT)),
                     proj_on_S=float(dot / nS), carry_act_linear=float(dot / nT))
# prefix per-prompt norms (constant) for one prompt
Sn, Sr, Tn, Tr = (bf(mm[k][0:1, 0]) for k in ('Sn', 'Sr', 'Tn', 'Tr'))
out['pre_one_prompt_norm_S'] = float(np.linalg.norm(Sr[0, 1:8] - Sn[0, 1:8])); out['pre_one_prompt_norm_T'] = float(np.linalg.norm(Tr[0, 1:8] - Tn[0, 1:8]))
out['pre_one_prompt_cos'] = float(((Sr - Sn)[0, 1:8] * (Tr - Tn)[0, 1:8]).sum() / out['pre_one_prompt_norm_S'] / out['pre_one_prompt_norm_T'])
out['rest_cos_per_prompt_median'] = float(np.median(cos_rest_pp)); out['rest_cos_per_prompt_p5_p95'] = [float(np.percentile(cos_rest_pp, 5)), float(np.percentile(cos_rest_pp, 95))]
out['bits_differ_train_none_vs_served_none'] = bits_TnSn / tot
out['bits_differ_train_ref_vs_train_none'] = bits_TrTn / tot
out['bits_differ_served_ref_vs_served_none'] = bits_SrSn / tot
# random-direction cosine scale on the prefix (7*1024 dims)
out['random_cos_sd_prefix'] = float(1 / np.sqrt(7 * 1024))
json.dump(out, open(sys.argv[0].replace('.py', '.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

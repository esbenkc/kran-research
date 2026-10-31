"""Vera: scout clause 3 from build's raw first-token scores (data/ft/*.json, per-prompt log-odds of an I/Sorry
opening), with MY cos(w_j, r) from the weights and my r. Change = mean(cell) - mean(base) on the same prompts.
Spearman(|cos|, |change|) over gate + 100 RN at pin 30; split-half reliability (even vs odd prompts)."""
import json, glob, re, numpy as np
from scipy.stats import spearmanr
import vera_fwd as F

D = '/Users/esben/work/theory-1031-refusal-dose-law/swarm-theory-1031-refusal-dose-law/data/ft/'
rh = np.load('vera_r13.npy')
W = F.T('model.layers.13.mlp.down_proj.weight').numpy()
base = np.array(json.load(open(D + 'base.json'))['score'])
units = {'G': 3270}
for f in glob.glob(D + 'RN_*@30.json'):
    units['RN_' + re.findall(r'RN_(\d+)@30', f)[0]] = int(re.findall(r'RN_(\d+)@30', f)[0])
cos, ch, ch_e, ch_o = [], [], [], []
for name, j in units.items():
    s = np.array(json.load(open(D + f'{name}@30.json'))['score'])
    b = base[:len(s)]
    cos.append(float(W[:, j] @ rh / np.linalg.norm(W[:, j])))
    d = s - b
    ch.append(d.mean()); ch_e.append(d[0::2].mean()); ch_o.append(d[1::2].mean())
cos, ch = np.array(cos), np.array(ch)
print('n units', len(units), 'n prompts', len(s))
print('split-half Spearman |change| even vs odd', spearmanr(np.abs(ch_e), np.abs(ch_o))[0])
print('c3 Spearman(|cos|, |change|)', spearmanr(np.abs(cos), np.abs(ch))[0])
print('signed Spearman(cos, change)', spearmanr(cos, ch)[0])
rn = slice(1, None)
print('RN only |.|', spearmanr(np.abs(cos[rn]), np.abs(ch[rn]))[0])
order = np.argsort(ch)
names = list(units)
print('largest drops', [(names[i], round(ch[i], 2), round(cos[i], 3)) for i in order[:5]])

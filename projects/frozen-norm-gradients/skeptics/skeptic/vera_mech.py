"""Mechanism numbers (Tove #15179), Qwen3 set A, n=200, from my own vectors vera_vec_qwen3_A.pt.
E = g_frz - g_real (gradient w.r.t. mlp_out_L[t], i.e. the residual after block L at t).
x_L = residual after block L at t (HF hidden_states[L+1][0, t]); y = centred logit of the true next token.
Reports per status layer: Pearson(E.x_L, y); median E.x_L / y; median cos(E, x_L); null: cos(E_i, x_L[j]) with a fixed
derangement j (default_rng(7)); also g_real.x_L / y and g_frz.x_L / y (how much of the score a pure scale-up of x_L
moves in each gradient)."""
import json, torch, numpy as np
from datasets import load_dataset
import vera_hf as V
from vera_stats_rms import derangement
E = V.HF('qwen3'); ds = load_dataset('NeelNanda/pile-10k', split='train')
rows = [json.loads(l) for l in open('vera_rows_qwen3_A.jsonl')]
vec = torch.load('vera_vec_qwen3_A.pt')
Ls = [19, 21, 23]
X = {L: [] for L in Ls}; Y = []
with torch.no_grad():
    for r in rows:
        ids = E.ids(ds[r['doc']]['text']); t = r['t']
        o = E.m(ids[None], output_hidden_states=True)
        lo = o.logits[0, t]; Y.append(float(lo[int(ids[t + 1])] - lo.mean()))
        for L in Ls: X[L].append(o.hidden_states[L + 1][0, t].double())
Y = np.array(Y); out = {'n': len(Y), 'median_y': float(np.median(Y))}
p = derangement(len(Y))
for L in Ls:
    x = torch.stack(X[L]); gr = vec[L]['real'].double(); gf = vec[L]['frz'].double(); Er = gf - gr
    ex = (Er * x).sum(1).numpy()
    cs = torch.nn.functional.cosine_similarity
    out[L] = {'pearson_Ex_y': float(np.corrcoef(ex, Y)[0, 1]),
              'median_Ex_over_y': float(np.median(ex / Y)),
              'median_greal_x_over_y': float(np.median((gr * x).sum(1).numpy() / Y)),
              'median_gfrz_x_over_y': float(np.median((gf * x).sum(1).numpy() / Y)),
              'median_cos_E_x': float(np.median(cs(Er, x, 1).numpy())),
              'null_median_cos_E_xderanged': float(np.median(cs(Er, x[p], 1).numpy())),
              'median_abs_cos_E_x': float(np.median(np.abs(cs(Er, x, 1).numpy()))),
              'null_median_abs_cos_E_xderanged': float(np.median(np.abs(cs(Er, x[p], 1).numpy())))}
    print(L, out[L], flush=True)
json.dump(out, open('vera_mech.json', 'w'), indent=1)

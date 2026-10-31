"""Vera (own code): clean-pass quantities for the dose number p on TEST 200, from my own CPU fp32 Qwen3 forward
(vera_fwd.py), stopped after layer 13. Per prompt, per prompt token: activation a_t of the gate and of the 6 natural
neurons, the L13-output residual norm ||h_t|| and h_t . r_hat. r_hat = my own DoM (vera_r13.npy, vera_dom.py).
Output: vera_clean.npz (object arrays, one entry per prompt).
"""
import json, time, numpy as np, torch
from transformers import AutoTokenizer
import vera_fwd as F

torch.set_num_threads(6)
S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
tok = AutoTokenizer.from_pretrained(F.SNAP)
NEUR = [3270, 1265, 3990, 4786, 2142, 5861, 2591]
rh = torch.from_numpy(np.load('vera_r13.npy'))
prompts = S['test_harmful']
ids = [F.encode(tok, p) for p in prompts]
xs = [F.T_rows('model.embed_tokens.weight', i) for i in ids]
pos = [torch.arange(len(i)) for i in ids]
acts, hn, hr = [], [], []
t0 = time.time()
for l in range(14):
    W = F.load_layer(l)
    for b in range(len(ids)):
        if l < 13:
            xs[b], _, _ = F.layer_step(W, xs[b], pos[b], None, None)
        else:
            # layer 13 by hand to keep all MLP activations
            x = xs[b]
            h = F.rms(x, W['input_layernorm'])
            q = (h @ W['self_attn.q_proj'].T).view(-1, F.NH, F.HD).transpose(0, 1)
            k = (h @ W['self_attn.k_proj'].T).view(-1, F.NKV, F.HD).transpose(0, 1)
            v = (h @ W['self_attn.v_proj'].T).view(-1, F.NKV, F.HD).transpose(0, 1)
            q, k = F.rms(q, W['self_attn.q_norm']), F.rms(k, W['self_attn.k_norm'])
            q, k = F.rope(q, pos[b]), F.rope(k, pos[b])
            kk, vv = k.repeat_interleave(F.NH // F.NKV, 0), v.repeat_interleave(F.NH // F.NKV, 0)
            s = q @ kk.transpose(1, 2) / F.math.sqrt(F.HD)
            T = s.shape[1]
            s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
            a = (s.softmax(-1) @ vv).transpose(0, 1).reshape(-1, F.NH * F.HD)
            x = x + a @ W['self_attn.o_proj'].T
            h2 = F.rms(x, W['post_attention_layernorm'])
            act = torch.nn.functional.silu(h2 @ W['mlp.gate_proj'].T) * (h2 @ W['mlp.up_proj'].T)
            out = x + act @ W['mlp.down_proj'].T
            acts.append(act[:, NEUR].numpy().astype(np.float32))
            hn.append(out.norm(dim=1).numpy())
            hr.append((out @ rh).numpy())
    del W
print('fwd', round(time.time() - t0, 1), 's')
np.savez('vera_clean.npz', acts=np.array(acts, dtype=object), hn=np.array(hn, dtype=object),
         hr=np.array(hr, dtype=object), neur=np.array(NEUR), ntok=np.array([len(i) for i in ids]))
a0 = np.concatenate([a[1:, 0] for a in acts])
print('gate a_t (t>=1): mean', a0.mean(), 'p5/50/95', np.quantile(a0, [.05, .5, .95]), 'max', a0.max())
print('||h_t|| t>=1 median', np.median(np.concatenate([h[1:] for h in hn])), 'last-token mean', np.mean([h[-1] for h in hn]))

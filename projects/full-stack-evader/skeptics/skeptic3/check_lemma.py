import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

with open('data/ref/split.json') as f:
    splits = json.load(f)

eval_false_idx = [i for i, x in enumerate(splits) if x['split'] == 'eval' and x['true'] == 0]
rng = np.random.RandomState(1021)
sample_idx = rng.choice(eval_false_idx, 150, replace=False)

mon = np.load('data/fits/panel.npz')
w = mon['tok29__w'].astype(np.float32)
sd = mon['tok29__sd'].astype(np.float32)
w_scaled = (w / sd)

refit_true_idx = [i for i, x in enumerate(splits) if x['split'] == 'refit' and x['true'] == 1]
clean_s = np.load('data/scores/clean.npz')['tok29']
line = np.sort(clean_s[refit_true_idx])[-2]

# false statements that are CAUGHT have score > line. We want to push them down by at least score - line
gaps = clean_s[sample_idx] - line
gaps = np.maximum(0, gaps)

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
ids = tok([r["text"] for r in splits], add_special_tokens=False)['input_ids']
stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
cpos = np.array([len(x) - 2 if s else len(x) - 1 for x, s in zip(ids, stop_own)])
off = np.load('data/acts/clean/off.npy')
c_rows = off[:-1] + cpos - 1

T = np.load('data/acts/clean/tok.npy', mmap_mode='r')
acts_sample = T[:, 29, :][c_rows][sample_idx].astype(np.float32)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32)
W = model.lm_head.weight.detach().numpy()

KL_needed_list = []
for i, act in enumerate(acts_sample):
    if gaps[i] <= 0:
        continue
    logits = act @ W.T
    topk = np.argsort(logits)[-4096:]
    logits_k = logits[topk]
    W_k = W[topk]
    
    logits_k_shift = logits_k - np.max(logits_k)
    exp = np.exp(logits_k_shift)
    p = exp / np.sum(exp)
    
    p_W = p[:, None] * W_k
    F = W_k.T @ p_W - (np.sum(p_W, axis=0, keepdims=True).T @ np.sum(p_W, axis=0, keepdims=True))
    F += np.eye(1536) * 1e-6
    F_inv = np.linalg.inv(F)
    val = (w_scaled @ F_inv @ w_scaled.T).item()
    
    kl_needed = (gaps[i]**2) / (2 * val)
    KL_needed_list.append(kl_needed)

kl_np = np.array(KL_needed_list)
print(f"Country token KL needed: median {np.median(kl_np):.2e}, max {np.max(kl_np):.2e}, frac > 0.001: {np.mean(kl_np > 0.001):.3f}")

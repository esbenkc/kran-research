"""Vera's own estimate of the layer-3 common-offset true Fisher of Pythia-70M (independent windows, own code).

F = E_x E_{y~p(.|x)} [g g'] / T,  g = d/d delta  sum_s log p(y_s | x, h3 + delta),
delta one 512-vector added to the output of block index 2 (= hidden_states[3]) at every position.
Windows: my own draw (seed 123) of 64-token windows from ODD pile-10k docs.
"""
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM

N_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
DRAWS, T, B = 2, 64, 16
dev = "cpu"  # MPS matmul assertion with autograd on this torch build
torch.manual_seed(123)
rng = np.random.default_rng(123)
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision="step143000")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", revision="step143000", dtype=torch.float32).to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)

ds = load_dataset("NeelNanda/pile-10k", split="train")
odd = [i for i in range(1, len(ds), 2)]
rng.shuffle(odd)
wins = []
for i in odd:
    ids = tok(ds[i]["text"])["input_ids"]
    if len(ids) < T + 1:
        continue
    s = rng.integers(0, len(ids) - T)
    wins.append(ids[s:s + T])
    if len(wins) == N_WIN:
        break
wins = torch.tensor(wins)
print("windows", wins.shape, "device", dev)

state = {}


def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    h = h + state["delta"]  # (B,1,512) broadcast over positions
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


model.gpt_neox.layers[2].register_forward_hook(hook)
F = torch.zeros(512, 512, dtype=torch.float64)
t0 = time.time()
n = 0
for b0 in range(0, len(wins), B):
    x = wins[b0:b0 + B].to(dev)
    with torch.no_grad():
        state["delta"] = torch.zeros(x.shape[0], 1, 512, device=dev)
        probs = torch.softmax(model(x).logits.float(), -1)
    for d in range(DRAWS):
        y = torch.distributions.Categorical(probs=probs).sample()  # (B,T)
        delta = torch.zeros(x.shape[0], 1, 512, device=dev, requires_grad=True)
        state["delta"] = delta
        lp = torch.log_softmax(model(x).logits.float(), -1).gather(-1, y[..., None]).squeeze(-1).sum()
        (g,) = torch.autograd.grad(lp, delta)
        g = g.squeeze(1).double().cpu()
        F += g.T @ g / DRAWS
    n += x.shape[0]
F = (F / n / T).numpy()
np.savez("vera_F.npz", F=F, n_win=n, draws=DRAWS)
print(f"done {n} windows in {time.time()-t0:.0f}s; trace {np.trace(F):.4f}, median eig {np.median(np.linalg.eigvalsh(F)):.5f}")

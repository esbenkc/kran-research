"""Own sanity: TC reconstruction FVU on own text + neuron-feature counts from weights."""
import torch, json
from vera_lib import *

torch.set_grad_enabled(False)
m, tok = load_gpt2()
texts = ["The quick brown fox jumps over the lazy dog. It was a sunny day in the city of London, and people were walking.",
         "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # prints 5",
         "In 1905 Albert Einstein published four papers that changed physics, including special relativity.",
         "The stock market fell sharply on Tuesday after the central bank raised interest rates by half a point."]
caps = {}
def mk(L):
    def h(mod, inp):
        caps[L] = inp[0].detach()
    return h
for L in range(8, 12):
    m.transformer.h[L].ln_2.register_forward_pre_hook(mk(L))
out = {}
for L in range(8, 12):
    tc = load_tc(L)
    Win, bin_, Wout, bout = mlp_in_z(m, L)
    Z, R, T = [], [], []
    for t in texts:
        ids = tok(t, return_tensors='pt').input_ids
        m(ids)
        x = caps[L][0]
        z = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
        real = gelu_new(z @ Win + bin_) @ Wout + bout
        a, rec = tc_forward(tc, z)
        R.append(real[1:]); T.append(rec[1:]); Z.append(a[1:])
    R, T, A = torch.cat(R), torch.cat(T), torch.cat(Z)
    fvu = ((R - T) ** 2).sum() / ((R - R.mean(0)) ** 2).sum()
    # cosines
    e = tc['W_enc']  # [768, F]
    d = tc['W_dec']  # [F, 768]
    en = e / e.norm(dim=0, keepdim=True); wn = Win / Win.norm(dim=0, keepdim=True)
    ce = en.T @ wn  # [F, 3072]
    dn = d / d.norm(dim=1, keepdim=True); on = Wout / Wout.norm(dim=1, keepdim=True)
    cd = dn @ on.T  # [F, 3072]
    emax, earg = ce.max(1); dmax, darg = cd.max(1)
    both = torch.minimum(ce, cd).max(1).values
    out[L] = dict(fvu=float(fvu), L0=float((A > 0).float().sum(1).mean()),
                  n_enc90=int((emax >= .9).sum()), n_dec90=int((dmax >= .9).sum()),
                  n_both90_same=int((both >= .9).sum()), n_both70_same=int((both >= .7).sum()),
                  n_maxcos90_either=int((torch.maximum(emax, dmax) >= .9).sum()),
                  frac_earg_eq_darg=float((earg == darg).float().mean()))
    print(L, out[L], flush=True)
json.dump(out, open('vera_sanity.json', 'w'), indent=1)

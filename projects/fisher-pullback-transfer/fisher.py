"""Output Fisher pulled back to layer 3, and the activation covariance, on clean pile-10k text.

Data: odd-index pile-10k documents (even ones feed the probes), first 64 tokens, docs with >= 64 tokens,
in a seed-0 random order. Windows 0..1999 -> Fisher; 2000..3999 -> retain-train (attack retain loss, Sigma);
4000..4499 -> retain-heldout (retain KL logged at the stop step). Saved to data/fisher/windows.npz.

Fisher (true Fisher: y sampled from the model, not the data labels). For each window, add a zero offset
delta_t at every position of hidden_states[3]; sample y_t ~ p_model(. | x_<=t) for all t; take
g_t = d/d delta_t  sum_t' log p(y_t').  One backward gives every g_t.
  F_common = E_seq[(sum_t g_t)(sum_t g_t)'] / T   (cost of ONE offset shared by all positions; primary)
  F_token  = E_seq E_t[g_t g_t']                     (per-position block-diagonal; sensitivity only)
With N_Y samples of y per window. In expectation over y, E[g g'] = J' F_out J (pull-back of the softmax Fisher).

Sigma: covariance of the mean-pooled (positions 1..63) layer-3 activations of the retain-train windows;
saved both as the sample covariance and Ledoit-Wolf shrunk (Ledoit-Wolf is the frozen one).
Run: uv run python projects/fisher-pullback-transfer/fisher.py
"""
import json
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.covariance import LedoitWolf

from common import DATA, LAYER, MAXTOK, device, load_model, pool, pooled_acts
from fetch_data import PILE

SEED = 0
N_FISHER, N_RETAIN, N_HELDOUT = 2000, 2000, 500
N_Y = 2
BS = 16


def pile_windows(tok, n=N_FISHER + N_RETAIN + N_HELDOUT, seed=SEED):
    ds = Dataset.from_file(str(PILE))
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(1, len(ds), 2))
    out = []
    for i in order:
        ids = tok(ds[int(i)]["text"], truncation=True, max_length=MAXTOK)["input_ids"]
        if len(ids) == MAXTOK:
            out.append((int(i), ids))
        if len(out) == n:
            break
    return out


def fisher_from_windows(model, ids, dev, t0):
    """F_common, F_token, the two split-half F_common, and the pooled activations of the windows."""
    holder = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + holder["delta"]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = model.gpt_neox.layers[LAYER - 1].register_forward_hook(hook)
    d = model.config.hidden_size
    Fc = torch.zeros(d, d, dtype=torch.float64)
    Ft = torch.zeros(d, d, dtype=torch.float64)
    Fc_half = torch.zeros(d, d, dtype=torch.float64)  # first half of the windows only (split-half reliability)
    P = np.zeros((len(ids), d), dtype=np.float32)
    n_g = 0
    for s in range(0, len(ids), BS):
        inp = ids[s:s + BS].to(dev)
        mask = torch.ones_like(inp, dtype=torch.float32)
        for r in range(N_Y):
            holder["delta"] = torch.zeros(inp.shape[0], MAXTOK, d, device=dev, requires_grad=True)
            out = model(input_ids=inp, output_hidden_states=(r == 0))
            logp = torch.log_softmax(out.logits, -1)
            with torch.no_grad():
                y = torch.multinomial(logp.exp().reshape(-1, logp.shape[-1]), 1).reshape(inp.shape)
            ll = logp.gather(-1, y[..., None]).sum()
            (g,) = torch.autograd.grad(ll, holder["delta"])
            g = g.cpu().double()  # (B,T,d)
            gs = g.sum(1)
            Fc += gs.T @ gs / MAXTOK
            if s < len(ids) // 2:
                Fc_half += gs.T @ gs / MAXTOK
            Ft += torch.einsum("btd,bte->de", g, g) / MAXTOK
            n_g += inp.shape[0]
            if r == 0:
                P[s:s + BS] = pool(out.hidden_states[LAYER].detach(), mask).cpu().numpy()
        if s % 800 == 0:
            print(f"{s}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    handle.remove()
    n_half = N_Y * (((len(ids) // 2) + BS - 1) // BS * BS)  # windows counted in the first-half accumulator
    Fc_h1 = (Fc_half / n_half).numpy()
    Fc_h2 = ((Fc - Fc_half) / (n_g - n_half)).numpy()
    return (Fc / n_g).numpy(), (Ft / n_g).numpy(), Fc_h1, Fc_h2, P


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    dev = device()
    tok, model = load_model(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    win = pile_windows(tok)
    (DATA / "fisher").mkdir(parents=True, exist_ok=True)
    doc = np.array([d for d, _ in win])
    allids = np.array([w for _, w in win])
    a, b = N_FISHER, N_FISHER + N_RETAIN
    np.savez_compressed(DATA / "fisher" / "windows.npz", fisher_ids=allids[:a], fisher_doc=doc[:a],
                        retain_ids=allids[a:b], retain_doc=doc[a:b], heldout_ids=allids[b:], heldout_doc=doc[b:])
    ids = torch.tensor(allids[:a])

    Fc, Ft, Fc_h1, Fc_h2, P = fisher_from_windows(model, ids, dev, t0)
    R = pooled_acts(model, [list(x) for x in allids[N_FISHER:N_FISHER + N_RETAIN]], tok.eos_token_id, dev)
    S = np.cov(R.T)
    lw = LedoitWolf().fit(R)
    ev_c, evec_c = np.linalg.eigh(Fc)
    ev_t, ev_s = (np.linalg.eigvalsh(M) for M in (Ft, S))
    np.savez_compressed(DATA / "fisher" / "fisher_sigma.npz", F_common=Fc, F_token=Ft, F_common_half1=Fc_h1,
                        F_common_half2=Fc_h2, F_common_evals=ev_c, F_common_evecs=evec_c, Sigma=S,
                        Sigma_LW=lw.covariance_, LW_shrinkage=lw.shrinkage_, pooled_retain=R, pooled_fisher=P,
                        mu_retain=R.mean(0))
    info = {"n_seq_fisher": len(ids), "n_seq_sigma": len(R), "n_y_per_seq": N_Y, "T": MAXTOK, "device": str(dev), "seconds": time.time() - t0,
            "lw_shrinkage": float(lw.shrinkage_), "F_common_n_eig_below_1e-8": int((ev_c < 1e-8).sum()),
            "F_common": {"trace": float(ev_c.sum()), "median_eig": float(np.median(ev_c)), "max_eig": float(ev_c.max()),
                         "min_eig": float(ev_c.min()), "eff_rank": float(ev_c.sum() ** 2 / (ev_c ** 2).sum())},
            "F_token": {"trace": float(ev_t.sum()), "median_eig": float(np.median(ev_t)), "max_eig": float(ev_t.max()),
                        "min_eig": float(ev_t.min()), "eff_rank": float(ev_t.sum() ** 2 / (ev_t ** 2).sum())},
            "Sigma": {"trace": float(ev_s.sum()), "median_eig": float(np.median(ev_s)), "max_eig": float(ev_s.max()),
                      "min_eig": float(ev_s.min()), "eff_rank": float(ev_s.sum() ** 2 / (ev_s ** 2).sum())}}
    (DATA / "fisher" / "fisher_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

"""The LEACE price, v1.3 s5 + D6: INDEPENDENT per-readout erasers (1027's all_indep rule), clean model, forward only.

LEACE (Belrose et al. 2023, arXiv 2306.03819; EleutherAI concept-erasure 0.2.4, LeaceFitter library defaults:
shrinkage on, constrain_cov_trace on) for one binary concept at each readout R = 10..29, each fit on the CLEAN
answer-position states of the fit pool (line + attack items, data/states/clean<TAG>.npy, one capture), then all 20
applied together at the answer position (last prompt token) in one forward pass.

  blind cell  : the eraser is fit on Z = h B, B = an orthonormal basis of the complement of U (data/ref/U_lens.npy),
                and applied inside the complement: h -> h + B (r(Z) - Z). The edit never touches U.
  full cell   : the eraser is fit and applied on h itself.
  shuf_<c>_<k>: the same fit on permutation k (rng 1028 + k, k = 0..4) of concept c's labels (null N(c)).

Price of a cell, eval prompts: P_ans = mean KL(p_clean || p_erased) of the next-token distribution at the answer
position (primary). Because only the last position is edited, every earlier position is unchanged, so the
every-position mean is exactly P_every = sum of the answer KLs / number of prompt tokens (reported beside it).
Q = P_blind(lie) / max(P_blind(decoy), N_blind(decoy)); premise also needs P_blind(lie) >= 10 N_blind(lie).
-> data/leace/price_blind<TAG>.json (merged per cell), data/leace/perprompt_<cell><TAG>.npy
Usage: ILL_TAG=_spread venv/bin/python price_blind.py [cells...]
"""
import json
import sys
import time

import numpy as np
import torch
from concept_erasure import LeaceFitter

from common import DATA, PANEL, REF, TAG, device, item_arrays, load_model, pad, tokenize

OUT = DATA / "leace"
DEFAULT = (["lie_blind", "decoy_blind"] + [f"shuf_{c}_blind_{k}" for c in ("decoy", "lie") for k in range(5)]
           + ["lie_full", "decoy_full"])


class Edit:
    """Hooks at readouts 10..29: h[answer] -> f_R(h[answer]) for the active cell's map f_R (torch, fp32)."""

    def __init__(self, model):
        self.inner = model.model
        self.f, self.ap = {}, None
        mods = {R: self.inner.layers[R - 1] for R in PANEL if R <= 28} | {29: self.inner.norm}
        self.hds = [m.register_forward_hook(self._mk(R)) for R, m in mods.items()]

    def _mk(self, R):
        def hook(_m, _i, out):
            if R not in self.f:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            ar = torch.arange(h.shape[0], device=h.device)
            h[ar, self.ap] = self.f[R](h[ar, self.ap].float()).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook


def main(cells):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    tok, model = load_model(dev)
    I, city, lie, decoy, split = item_arrays()
    ids = tokenize(tok, [r["text"] for r in I])
    H = np.load(DATA / "states" / f"clean{TAG}.npy", mmap_mode="r")
    U = np.load(REF / "U_lens.npy")
    d, k = U.shape
    Q_, _ = np.linalg.qr(np.concatenate([U, np.random.default_rng(0).standard_normal((d, d - k))], 1))
    B = Q_[:, k:]  # complement basis, d x (d - k)
    assert np.abs(U.T @ B).max() < 1e-4
    Bt = torch.tensor(B, dtype=torch.float32, device=dev)
    pool = np.where(np.isin(split, ["line", "attack"]))[0]
    ev = np.where(split == "eval")[0]
    labels = {"lie": lie, "decoy": decoy}
    E = Edit(model)
    pid = tok.pad_token_id
    log = lambda *a: print(*a, f"{time.time() - t0:.0f}s", flush=True)
    order = ev[np.argsort([len(ids[i]) for i in ev], kind="stable")]
    batches = [order[s:s + 32] for s in range(0, len(order), 32)]
    n_tok = int(sum(len(ids[i]) for i in ev))

    @torch.no_grad()
    def answer_logp(o):
        inp, mask = pad([ids[i] for i in o], pid, dev)
        ap = torch.tensor([len(ids[i]) - 1 for i in o], device=dev)
        E.ap = ap
        h = E.inner(input_ids=inp, attention_mask=mask).last_hidden_state
        return model.lm_head(h[torch.arange(len(o), device=dev), ap]).float().log_softmax(-1)

    E.f = {}
    clean = [answer_logp(o).cpu() for o in batches]
    log("clean pass done")

    def fit(y, blind):
        fs = {}
        for kk, R in enumerate(PANEL):
            X = torch.tensor(np.asarray(H[pool, kk], np.float32), dtype=torch.float64)
            if blind:
                X = X @ torch.tensor(B, dtype=torch.float64)
            e = LeaceFitter.fit(X, torch.tensor(y[pool], dtype=torch.float64)[:, None]).eraser
            L, Rt, b = e.proj_left.float().to(dev), e.proj_right.float().to(dev), e.bias.float().to(dev)
            if blind:
                fs[R] = (lambda L, Rt, b: lambda x: x - (((x @ Bt - b) @ Rt.mT) @ L.mT) @ Bt.mT)(L, Rt, b)
            else:
                fs[R] = (lambda L, Rt, b: lambda x: x - ((x - b) @ Rt.mT) @ L.mT)(L, Rt, b)
        return fs

    pf = OUT / f"price_blind{TAG}.json"
    res = json.loads(pf.read_text()) if pf.exists() else {}
    for cell in cells:
        p = cell.split("_")
        if p[0] == "shuf":
            y = np.random.default_rng(1028 + int(p[3])).permutation(labels[p[1]])
            blind = p[2] == "blind"
        else:
            y, blind = labels[p[0]], p[1] == "blind"
        E.f = fit(y, blind)
        per = np.zeros(len(I))
        for o, lp in zip(batches, clean):
            lq = answer_logp(o).cpu()
            per[o] = (lp.exp() * (lp - lq)).sum(-1).numpy()
        E.f = {}
        if dev.type == "mps":
            torch.mps.empty_cache()
        np.save(OUT / f"perprompt_{cell}{TAG}.npy", per[ev])
        res[cell] = {"P_answer": float(per[ev].mean()), "P_every_position": float(per[ev].sum() / n_tok),
                     "n_eval": len(ev), "n_tokens": n_tok, "fit_pool": len(pool), "k_lens": k}
        pf.write_text(json.dumps(res, indent=1))
        log(cell, json.dumps(res[cell]))
    for sp in ("blind", "full"):
        if f"lie_{sp}" in res and f"decoy_{sp}" in res:
            out = {"raw_ratio": res[f"lie_{sp}"]["P_answer"] / res[f"decoy_{sp}"]["P_answer"]}
            nd = [res[f"shuf_decoy_{sp}_{j}"]["P_answer"] for j in range(5) if f"shuf_decoy_{sp}_{j}" in res]
            nl = [res[f"shuf_lie_{sp}_{j}"]["P_answer"] for j in range(5) if f"shuf_lie_{sp}_{j}" in res]
            if len(nd) == 5 and len(nl) == 5:
                out |= {"N_decoy": float(np.mean(nd)), "N_lie": float(np.mean(nl)),
                        "Q": res[f"lie_{sp}"]["P_answer"] / max(res[f"decoy_{sp}"]["P_answer"], float(np.mean(nd))),
                        "lie_over_own_null": res[f"lie_{sp}"]["P_answer"] / float(np.mean(nl))}
                out["premise_Q_ge_10"] = bool(out["Q"] >= 10 and out["lie_over_own_null"] >= 10)
            res[f"Q_{sp}"] = out
            pf.write_text(json.dumps(res, indent=1))
            log(f"Q_{sp}", json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

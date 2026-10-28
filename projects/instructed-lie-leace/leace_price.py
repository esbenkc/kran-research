"""The LEACE price (PREREG s4), clean model, forward passes only.

LEACE (Belrose et al. 2023, arXiv 2306.03819; EleutherAI concept-erasure 0.2.4, LeaceFitter with library defaults =
shrinkage on, constrain_cov_trace on) is the affine edit x -> x - P (x - b) of smallest mean squared change that
makes a binary concept linearly unreadable (equal class-conditional means). The edit acts at the ANSWER POSITION
(last prompt token) only, at readouts R = 10..29 (R <= 28: output of block R-1; 29: final-norm output), fit
SEQUENTIALLY: the eraser at R is fit on states at R with the erasers at 10..R-1 already applied (1027 'seq 10..29').
Fit pool = line + attack items (kept cities, all 4 prompt types), the same items for every concept.

  P(concept)  = mean over EVERY position of the eval prompts of KL(p_clean || p_erased)   (nats per token, primary)
  P_ans       = mean over eval prompts of the same KL at the answer position only
Because the edit is at the last position, every earlier position is unchanged, and P = P_ans * n_eval / n_tokens.
Q = P(lie) / P(decoy) governs (PREREG s4/s6).

Cells (--cells, default 'lie decoy'):
  lie, decoy              : registered erasers (status pair)
  shuf_<c>_<k>            : the same fit with permutation k (rng 1028+k) of concept c's labels (null)
  rand_<c>                : concept c's registered rank-1 edits with the direction rotated to a fixed random unit vector
                            of the same length (rng 1028): same per-item size, no concept (null)
  everypos_<c>            : concept c's registered erasers applied at EVERY prompt position (Iris #9196 B)
  plain_<c>               : LeaceFitter(shrinkage=False), sequential, same pool (sensitivity)
-> data/leace/price.json (merged per cell), data/leace/erasers_<cell>.npz, data/leace/perprompt_<cell>.npy
Usage: venv/bin/python leace_price.py [cells...]
"""
import json
import sys
import time

import numpy as np
import torch
from concept_erasure import LeaceFitter

from common import DATA, PANEL, TAG, device, item_arrays, load_model, pad, tokenize

OUT = DATA / "leace"


class Stop(Exception):
    pass


class Er:
    def __init__(self, e=None, dev=None, arrays=None):
        if arrays is not None:
            self.L, self.R, self.b = (torch.tensor(arrays[k], device=dev) for k in ("L", "R", "b"))
            return
        self.L, self.R, self.b = e.proj_left.float().to(dev), e.proj_right.float().to(dev), e.bias.float().to(dev)

    def __call__(self, x):
        return x - ((x - self.b) @ self.R.mT) @ self.L.mT

    def arrays(self):
        return {"L": self.L.cpu().numpy(), "R": self.R.cpu().numpy(), "b": self.b.cpu().numpy()}


class RandDir:
    """x - ((x - b) R^T) u^T with u = a fixed random unit vector scaled to |L| (rank-1 edit, same size)."""

    def __init__(self, e, v):
        self.e, self.u = e, (v * e.L.norm())[:, None]

    def __call__(self, x):
        return x - ((x - self.e.b) @ self.e.R.mT) @ self.u.mT


class Edit:
    """Forward hooks at readouts 10..29: apply er[R] at the answer position (or every position), capture the edited
    answer state at readout `cap`, and optionally stop the forward right after it (fit passes)."""

    def __init__(self, model):
        self.inner, self.lm_head = model.model, model.lm_head
        self.er, self.cap, self.stop, self.everypos, self.ap, self.box = {}, None, False, False, None, None
        mods = {R: self.inner.layers[R - 1] for R in PANEL if R <= 28} | {29: self.inner.norm}
        self.hds = [m.register_forward_hook(self._mk(R)) for R, m in mods.items()]

    def _mk(self, R):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            ar = torch.arange(h.shape[0], device=h.device)
            edit = R in self.er
            if edit:
                h = h.clone()
                if self.everypos:
                    B, T, d = h.shape
                    h = self.er[R](h.float().reshape(-1, d)).reshape(B, T, d).to(h.dtype)
                else:
                    h[ar, self.ap] = self.er[R](h[ar, self.ap].float()).to(h.dtype)
            if R == self.cap:
                self.box = h[ar, self.ap].float()
                if self.stop:
                    raise Stop
            if edit:
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook


def main(cells):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    tok, model = load_model(dev)
    I, city, lie, decoy, split = item_arrays()
    ids = tokenize(tok, [r["text"] for r in I])
    pid = tok.pad_token_id
    E = Edit(model)
    pool = np.where(np.isin(split, ["line", "attack"]))[0]
    ev = np.where(split == "eval")[0]
    labels = {"lie": lie, "decoy": decoy}
    log = lambda *a: print(*a, f"{time.time() - t0:.0f}s", flush=True)

    def batches(idx, bs=32):
        order = idx[np.argsort([len(ids[i]) for i in idx], kind="stable")]
        for s in range(0, len(order), bs):
            o = order[s:s + bs]
            inp, mask = pad([ids[i] for i in o], pid, dev)
            yield o, inp, mask, torch.tensor([len(ids[i]) - 1 for i in o], device=dev)

    @torch.no_grad()
    def states_at(R):
        E.cap, E.stop = R, True
        X, order = [], []
        for o, inp, mask, ap in batches(pool):
            E.ap = ap
            try:
                E.inner(input_ids=inp, attention_mask=mask)
            except Stop:
                pass
            X.append(E.box.cpu()); order.append(o)
        E.cap, E.stop = None, False
        o = np.concatenate(order)
        X = torch.cat(X).double()
        inv = np.empty_like(o); inv[np.searchsorted(np.sort(pool), o)] = np.arange(len(o))
        return X[inv], np.sort(pool)

    def fit_seq(y, kw):
        E.er = {}
        ers = {}
        for R in PANEL:
            X, rows = states_at(R)
            f = LeaceFitter.fit(X, torch.tensor(y[rows], dtype=torch.float64)[:, None], **kw)
            ers[R] = Er(f.eraser, dev)
            E.er = dict(ers)
        E.er = {}
        return ers

    @torch.no_grad()
    def price(ers, everypos=False):
        tot, n_tok, ans = 0.0, 0, np.zeros(len(ids))
        for o, inp, mask, ap in batches(ev):
            E.er, E.everypos, E.ap = {}, False, ap
            lp = model.lm_head(E.inner(input_ids=inp, attention_mask=mask).last_hidden_state).float().log_softmax(-1)
            E.er, E.everypos = ers, everypos
            lq = model.lm_head(E.inner(input_ids=inp, attention_mask=mask).last_hidden_state).float().log_softmax(-1)
            E.er, E.everypos = {}, False
            kl = (lp.exp() * (lp - lq)).sum(-1) * mask
            tot += float(kl.sum()); n_tok += int(mask.sum())
            ans[o] = kl[torch.arange(len(o), device=dev), ap].cpu().numpy()
            del lp, lq, kl
        if dev.type == "mps":
            torch.mps.empty_cache()
        return {"P_every_position": tot / n_tok, "P_answer": float(ans[ev].mean()), "n_eval": len(ev), "n_tokens": n_tok}, ans[ev]

    pf = OUT / f"price{TAG}.json"
    res = json.loads(pf.read_text()) if pf.exists() else {}
    reg = {}
    for cell in cells:
        parts = cell.split("_")
        if cell in labels:
            ers = fit_seq(labels[cell], {})
        elif parts[0] == "shuf":
            y = np.random.default_rng(1028 + int(parts[2])).permutation(labels[parts[1]])
            ers = fit_seq(y, {})
        elif parts[0] == "plain":
            ers = fit_seq(labels[parts[1]], {"shrinkage": False})
        elif parts[0] in ("rand", "everypos"):
            Z = np.load(OUT / f"erasers_{parts[1]}{TAG}.npz")
            base = {R: Er(None, dev, {a: Z[f"R{R}__{a}"] for a in ("L", "R", "b")}) for R in PANEL}
            if parts[0] == "rand":
                g = torch.Generator().manual_seed(1028)
                ers = {}
                for R in PANEL:
                    v = torch.randn(base[R].L.shape[0], generator=g)
                    ers[R] = RandDir(base[R], (v / v.norm()).to(dev))
            else:
                ers = base
        else:
            raise ValueError(cell)
        p, per = price(ers, everypos=cell.startswith("everypos"))
        p["edit_norm_mean"] = None
        if all(isinstance(e, Er) for e in ers.values()) and parts[0] not in ("rand", "everypos"):
            np.savez(OUT / f"erasers_{cell}{TAG}.npz", **{f"R{R}__{k}": v for R, e in ers.items() for k, v in e.arrays().items()})
        np.save(OUT / f"perprompt_{cell}{TAG}.npy", per)
        res[cell] = p
        pf.write_text(json.dumps(res, indent=1))
        log(cell, json.dumps(p))
    if "lie" in res and "decoy" in res:
        res["Q"] = {u: res["lie"][u] / res["decoy"][u] for u in ("P_every_position", "P_answer")}
        pf.write_text(json.dumps(res, indent=1))
        log("Q", json.dumps(res["Q"]))


if __name__ == "__main__":
    main(sys.argv[1:] or ["lie", "decoy"])

"""Country-token activations at the PANEL readouts + per-token log-probs + per-token output KL vs clean, for one
model state (ported from the 1.5B panel-fine-tune extract.py; at 8B only the country token at the panel readouts is
stored, ~0.5 GB per state instead of ~9 GB).

Output: data/acts/<state>/cty.npy fp16 (2992, P, d) + readouts.json (not committed; sha256 in data/acts_sha256.txt);
data/states/<state>/ (committed):
  off.npy  int64 (2993,): statement i owns rows off[i]:off[i+1] of the flat per-token arrays (tokens 1..n-1).
  lp.npy   float32 (n_tok,): log P(token | earlier tokens) of every row (float32 log-softmax).
  kl_pos0.npy float32 (n_stmt,): attacked states only: the same KL at position 0 (predicting token 1).
  kl.npy   float32 (n_tok,): attacked states only: KL(p_clean || p_state) of the next-token distribution AT that
           row's position (row r = position t >= 1 predicts token t+1), clean = same model with adapters disabled,
           same batch (float32 log-softmax).
  kl_pile.npy float32 (200, 64): attacked states only: the same per-token KL on the 200 held-out pile windows
           (all 64 positions).
  meta.json

Run: .venv/bin/python extract.py --state clean | --state <name> --adapter data/runs/<name>__adapter
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ACTS, DATA, PANEL, POSTNORM, REF, country_pos, device, load_model, load_split, pad, token_offsets, \
    tokenize


class Capture:
    """Forward hooks on the embedding and on every decoder block: box[R] (R = 0..28); the final-norm output is
    returned by run() (R = 29)."""

    def __init__(self, model, readouts):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.inner = base.model
        self.lm_head = base.lm_head
        self.box = {}
        self.hds = [self.inner.layers[R - 1].register_forward_hook(self._mk(R)) for R in readouts if R != POSTNORM]

    def _mk(self, R):
        def hook(_m, _i, out):
            self.box[R] = out[0] if isinstance(out, tuple) else out
        return hook

    def run(self, inp, mask):
        self.box = {}
        return self.inner(input_ids=inp, attention_mask=mask).last_hidden_state

    def remove(self):
        for h in self.hds:
            h.remove()


def kl_rows(lm_head, h_ref, h, nxt=None, chunk=8):
    """From post-norm states (B, T, d): per-position KL(p_ref || p) (B, T) float32 numpy (None if h_ref is None) and,
    if nxt (B, T-1) is given, lp[r, t-1] = log p(token t | < t) (B, T-1) float32 numpy."""
    kl, lp = [], []
    for c in range(0, h.shape[0], chunk):
        lq = lm_head(h[c:c + chunk]).float().log_softmax(-1)
        if h_ref is not None:
            lr = lm_head(h_ref[c:c + chunk]).float().log_softmax(-1)
            kl.append((lr.exp() * (lr - lq)).sum(-1).cpu().numpy())
            del lr
        if nxt is not None:
            lp.append(lq[:, :-1].gather(-1, nxt[c:c + chunk, :, None])[..., 0].cpu().numpy())
        del lq
    return (np.concatenate(kl) if kl else None), (np.concatenate(lp) if lp else None)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--all", action="store_true", help="store every readout 1..n+1 (clean: PREREG s8 best single)")
    a = ap.parse_args()
    t0 = time.time()
    dev = device()
    tok, model = load_model(dev, a.adapter)
    attacked = a.adapter is not None
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    off = token_offsets(ids)
    cpos, _ = country_pos(tok, ids)
    nB = model.config.num_hidden_layers
    RD = list(range(1, nB + 2)) if a.all else PANEL
    cap = Capture(model, RD)
    d = cap.inner.config.hidden_size
    out = ACTS / a.state
    out.mkdir(parents=True, exist_ok=True)
    sto = DATA / "states" / a.state
    sto.mkdir(parents=True, exist_ok=True)
    P = len(RD)
    T = np.zeros((len(ids), P, d), np.float16)
    LP = np.zeros(int(off[-1]), np.float32)
    KL = np.zeros(int(off[-1]), np.float32)
    KL0 = np.zeros(len(ids), np.float32)  # position 0 of every statement (PREREG: every position)
    maxabs = np.zeros(P)
    order = np.argsort([len(x) for x in ids])
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        inp, mask = pad([ids[i] for i in idx], tok.pad_token_id, dev)
        last = cap.run(inp, mask)
        cap.box[POSTNORM] = last
        ar = torch.arange(len(idx), device=dev)
        cp = torch.as_tensor(cpos[idx], device=dev)
        H = torch.stack([cap.box[R][ar, cp] for R in RD], 1).float().cpu().numpy()  # (B, P, d)
        maxabs = np.maximum(maxabs, np.abs(H).max((0, 2)))
        T[idx] = H.astype(np.float16)
        last_ref = None
        if attacked:
            with model.disable_adapter():
                last_ref = cap.run(inp, mask)
        klm, lp = kl_rows(cap.lm_head, last_ref, last, inp[:, 1:])
        for r, i in enumerate(idx):
            n = len(ids[i])
            LP[off[i]:off[i + 1]] = lp[r, :n - 1]
            if attacked:
                KL[off[i]:off[i + 1]] = klm[r, 1:n]
                KL0[i] = klm[r, 0]
        cap.box = {}
        del H, last, last_ref
        if dev.type == "mps":
            torch.mps.empty_cache()
    np.save(out / "cty.npy", T)
    (out / "readouts.json").write_text(json.dumps(RD))
    # PREREG s8: country-token state norms at the last pre-norm readout (n) and the post-norm readout (n+1)
    np.savez(sto / "country_norms.npz", **{f"R{R}": np.linalg.norm(T[:, RD.index(R)].astype(np.float32), axis=1)
                                           for R in (nB, nB + 1)})
    np.save(sto / "off.npy", off)
    np.save(sto / "lp.npy", LP)
    np.save(sto / "country_pos.npy", cpos)
    meta = {"state": a.state, "adapter": a.adapter, "n_stmt": len(ids), "n_tok": int(off[-1]), "readouts": RD,
            "n_layers": nB, "d": d, "batch": a.bs, "max_abs_act_per_readout": maxabs.round(2).tolist(),
            "fp16_overflow": bool(maxabs.max() >= 65504)}
    if attacked:
        np.save(sto / "kl.npy", KL)
        np.save(sto / "kl_pos0.npy", KL0)
        held = np.load(REF / "retain.npz")["heldout"]
        kp = []
        for s in range(0, len(held), 8):
            inp = torch.tensor(held[s:s + 8]).to(dev)
            last = cap.run(inp, torch.ones_like(inp))
            with model.disable_adapter():
                last_ref = cap.run(inp, torch.ones_like(inp))
            kp.append(kl_rows(cap.lm_head, last_ref, last)[0])
        kp = np.concatenate(kp)
        np.save(sto / "kl_pile.npy", kp)
        meta["kl_pile_mean"] = float(kp.mean())
        ev = np.array([r["split"] == "eval" for r in rows])
        evrows = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(ev)[0]])
        meta["kl_eval_mean_every_position"] = float((KL[evrows].sum() + KL0[ev].sum()) / (len(evrows) + ev.sum()))
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (sto / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "max_abs_act_per_readout"}), flush=True)


if __name__ == "__main__":
    main()

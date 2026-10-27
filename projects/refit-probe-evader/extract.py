"""Per-token activations at every readout + per-token log-probs + per-token output KL vs clean, for one model state.

Output (data/acts/<state>/, not committed; data/acts_sha256.txt):
  tok.npy  fp16 (n_tok, 30, 1536): row r = token r of the flat token list (tokens 1..n-1 of every statement, split
           order); readout R as in common.py (0 = embedding, 1..28 = block outputs before the final norm,
           29 = HF hidden_states[28] = after the final norm = the input of the logits).
  off.npy  int64 (2993,): statement i owns rows off[i]:off[i+1].
  lp.npy   float32 (n_tok,): log P(token | earlier tokens) of every row (float32 log-softmax).
  kl_pos0.npy float32 (n_stmt,): attacked states only: the same KL at position 0 (predicting token 1).
  kl.npy   float32 (n_tok,): attacked states only: KL(p_clean || p_state) of the next-token distribution AT that
           row's position (row r = position t >= 1 predicts token t+1), clean = same model with adapters disabled,
           same batch (float32 log-softmax).
  kl_pile.npy float32 (200, 64): attacked states only: the same per-token KL on the 200 held-out pile windows
           (all 64 positions).
  meta.json
Gate (clean only): readouts 0..28 must equal the 1020 clean memmap (megaprobe-after/data/acts/clean/tok.npy) exactly.

Run: .venv/bin/python extract.py --state clean | --state <name> --adapter data/runs/<name>__adapter
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ACTS, REF, device, load_model, load_split, pad, token_offsets, tokenize

REF1020 = Path.home() / "work/theory-1020-megaprobe-after/kran-research/projects/megaprobe-after/data/acts/clean/tok.npy"


class Capture:
    """Forward hooks on the embedding and on every decoder block: box[R] (R = 0..28); the final-norm output is
    returned by run() (R = 29)."""

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.inner = base.model
        self.lm_head = base.lm_head
        self.box = {}
        self.hds = [self.inner.embed_tokens.register_forward_hook(self._mk(0))]
        self.hds += [blk.register_forward_hook(self._mk(i + 1)) for i, blk in enumerate(self.inner.layers)]

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
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lean", action="store_true", help="no per-token memmap; save only country-token states "
                    "(data/ctok/<state>.npy, fp16, readouts 10..29) + lp/kl")
    a = ap.parse_args()
    t0 = time.time()
    dev = device()
    tok, model = load_model(dev, a.adapter)
    attacked = a.adapter is not None
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    off = token_offsets(ids)
    cap = Capture(model)
    nB = len(cap.inner.layers)
    d = cap.inner.config.hidden_size
    out = ACTS / a.state
    out.mkdir(parents=True, exist_ok=True)
    from common import CFG, DATA, country_pos
    PANEL = CFG["panel"]["readouts"]
    cpos, _ = country_pos(tok, ids)
    CTK = np.zeros((len(ids), len(PANEL), d), np.float16)
    T = None if a.lean else np.lib.format.open_memmap(out / "tok.npy", mode="w+", dtype=np.float16,
                                                      shape=(int(off[-1]), nB + 2, d))
    LP = np.zeros(int(off[-1]), np.float32)
    KL = np.zeros(int(off[-1]), np.float32)
    KL0 = np.zeros(len(ids), np.float32)  # position 0 of every statement (PREREG s6: every position)
    maxabs = np.zeros(nB + 2)
    order = np.argsort([len(x) for x in ids])
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        inp, mask = pad([ids[i] for i in idx], tok.pad_token_id, dev)
        last = cap.run(inp, mask)
        H = torch.stack([cap.box[R] for R in range(nB + 1)] + [last], 2)  # (B, T, 30, d) bf16
        Hc = H.float().cpu().numpy()
        maxabs = np.maximum(maxabs, np.abs(Hc).reshape(-1, nB + 2, d).max((0, 2)))
        last_ref = None
        if attacked:
            with model.disable_adapter():
                last_ref = cap.run(inp, mask)
        klm, lp = kl_rows(cap.lm_head, last_ref, last, inp[:, 1:])
        for r, i in enumerate(idx):
            LP[off[i]:off[i + 1]] = lp[r, :len(ids[i]) - 1]
        for r, i in enumerate(idx):
            n = len(ids[i])
            CTK[i] = Hc[r, cpos[i], PANEL].astype(np.float16)
            if T is not None:
                T[off[i]:off[i + 1]] = Hc[r, 1:n].astype(np.float16)
            if attacked:
                KL[off[i]:off[i + 1]] = klm[r, 1:n]
                KL0[i] = klm[r, 0]
        del H, Hc, last, last_ref
        if dev.type == "mps":
            torch.mps.empty_cache()
    if T is not None:
        T.flush()
    (DATA / "ctok").mkdir(exist_ok=True)
    np.save(DATA / "ctok" / f"{a.state}.npy", CTK)
    np.save(out / "off.npy", off)
    np.save(out / "lp.npy", LP)
    meta = {"state": a.state, "adapter": a.adapter, "n_stmt": len(ids), "n_tok": int(off[-1]), "n_readouts": nB + 2,
            "d": d, "batch": a.bs, "max_abs_act_per_readout": maxabs.round(2).tolist(),
            "fp16_overflow": bool(maxabs.max() >= 65504)}
    if attacked:
        np.save(out / "kl.npy", KL)
        np.save(out / "kl_pos0.npy", KL0)
        held = np.load(REF / "retain.npz")["heldout"]
        kp = []
        for s in range(0, len(held), 8):
            inp = torch.tensor(held[s:s + 8]).to(dev)
            last = cap.run(inp, torch.ones_like(inp))
            with model.disable_adapter():
                last_ref = cap.run(inp, torch.ones_like(inp))
            kp.append(kl_rows(cap.lm_head, last_ref, last)[0])
        kp = np.concatenate(kp)
        np.save(out / "kl_pile.npy", kp)
        meta["kl_pile_mean"] = float(kp.mean())
        ev = np.array([r["split"] == "eval" for r in rows])
        evrows = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(ev)[0]])
        meta["kl_eval_mean_every_position"] = float((KL[evrows].sum() + KL0[ev].sum()) / (len(evrows) + ev.sum()))
    elif T is not None:
        ref = np.load(REF1020, mmap_mode="r") if REF1020.exists() else None
        if ref is not None:
            mism = 0
            for s in range(0, int(off[-1]), 4096):
                mism += int((np.asarray(T[s:s + 4096, :nB + 1]) != np.asarray(ref[s:s + 4096])).sum())
            meta["gate_mismatches_vs_1020_readouts_0_28"] = mism
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "max_abs_act_per_readout"}), flush=True)


if __name__ == "__main__":
    main()

"""Per-token activations at EVERY layer + per-token log-probs, for one model state (clean or a 1019 lam0 adapter).

Output (data/acts/<state>/, not committed):
  tok.npy  fp16 (n_tok, 29, 1536): row r = token r of the flat token list (tokens 1..n-1 of every statement, 1019
           order); layer index L = hidden_states[L]: L=0 the token embedding, L>=1 the output of decoder block L-1
           (before the final RMSNorm, also for L=28).
  off.npy  int64 (2993,): statement i owns rows off[i]:off[i+1].
  lp.npy   float32 (n_tok,): log P(token | earlier tokens) of every row, from float32 log-softmax.
  meta.json: state, adapter, n_tok, per-layer max |act| (fp16 overflow check), gate numbers.
Gate (printed + meta.json): the layer-14 mean over tokens, scored with the frozen 1019 probe, must match the 1019 log
(clean: data/ref1019/clean_acts.npz score; seed s: data/ref1019/runs/lam0__seed<s>.npz scores[-1]).

Run: uv run python extract.py --state clean|seed0|seed1|seed2   (one model job at a time)
"""
import argparse
import json
import time

import numpy as np
import torch

from common import ACTS, ATTACK_LAYER, REF, adapter_path, device, load_model, load_probe, load_split, pad, \
    token_offsets, tokenize


class Capture:
    """Forward hooks on the embedding and on every decoder block: box[L] = hidden_states[L] (L = 0..n_blocks)."""

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.inner = base.model
        self.lm_head = base.lm_head
        self.box = {}
        self.hds = [self.inner.embed_tokens.register_forward_hook(self._mk(0))]
        self.hds += [blk.register_forward_hook(self._mk(i + 1)) for i, blk in enumerate(self.inner.layers)]

    def _mk(self, L):
        def hook(_m, _i, out):
            self.box[L] = out[0] if isinstance(out, tuple) else out
        return hook

    def run(self, inp, mask):
        self.box = {}
        out = self.inner(input_ids=inp, attention_mask=mask)
        return out.last_hidden_state  # = final RMSNorm(hidden_states[n_blocks])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, choices=["clean", "seed0", "seed1", "seed2"])
    ap.add_argument("--bs", type=int, default=64)  # 1019 batch size + length-sorted order -> identical L14 readout
    a = ap.parse_args()
    t0 = time.time()
    dev = device()
    tok, model = load_model(dev, adapter_path(a.state))
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    off = token_offsets(ids)
    cap = Capture(model)
    nL = len(cap.inner.layers) + 1
    d = cap.inner.config.hidden_size
    out = ACTS / a.state
    out.mkdir(parents=True, exist_ok=True)
    T = np.lib.format.open_memmap(out / "tok.npy", mode="w+", dtype=np.float16, shape=(int(off[-1]), nL, d))
    LP = np.zeros(int(off[-1]), np.float32)
    mean14 = np.zeros((len(ids), d), np.float64)  # gate: from float32 before the fp16 cast
    maxabs = np.zeros(nL)
    order = np.argsort([len(x) for x in ids])
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        inp, mask = pad([ids[i] for i in idx], tok.pad_token_id, dev)
        last = cap.run(inp, mask)
        H = torch.stack([cap.box[L] for L in range(nL)], 2)  # (B, T, nL, d) bf16
        Hc = H.float().cpu().numpy()
        maxabs = np.maximum(maxabs, np.abs(Hc).reshape(-1, nL, d).max((0, 2)))
        for c in range(0, len(idx), 8):  # log-probs in float32, 8 rows at a time (vocab 151,936)
            lg = cap.lm_head(last[c:c + 8]).float().log_softmax(-1)  # (b, T, V)
            nxt = inp[c:c + 8, 1:]
            lp = lg[:, :-1].gather(-1, nxt[..., None])[..., 0].cpu().numpy()  # lp[r, t-1] = log P(token t)
            for r in range(lp.shape[0]):
                i = idx[c + r]
                LP[off[i]:off[i + 1]] = lp[r, :len(ids[i]) - 1]
            del lg
        for r, i in enumerate(idx):
            n = len(ids[i])
            T[off[i]:off[i + 1]] = Hc[r, 1:n].astype(np.float16)
            mean14[i] = Hc[r, 1:n, ATTACK_LAYER].astype(np.float64).mean(0)
        del H, Hc, last
        if dev.type == "mps":
            torch.mps.empty_cache()
    T.flush()
    np.save(out / "off.npy", off)
    np.save(out / "lp.npy", LP)
    w, b = load_probe()
    s14 = mean14 @ w + b
    ref = np.load(REF / "clean_acts.npz")["score"] if a.state == "clean" else \
        np.load(REF / "runs" / f"lam0__seed{a.state[-1]}.npz")["scores"][-1]
    T16 = np.load(out / "tok.npy", mmap_mode="r")
    m16 = np.add.reduceat(np.asarray(T16[:, ATTACK_LAYER], np.float64), off[:-1], axis=0) / np.diff(off)[:, None]
    meta = {"state": a.state, "adapter": str(adapter_path(a.state)), "n_stmt": len(ids), "n_tok": int(off[-1]),
            "n_layers": nL, "d": d, "batch": a.bs, "max_abs_act_per_layer": maxabs.round(2).tolist(),
            "fp16_overflow": bool(maxabs.max() >= 65504),
            "gate_max_abs_score_diff_vs_1019_float32": float(np.abs(s14 - ref).max()),
            "gate_max_abs_score_diff_vs_1019_fp16_store": float(np.abs(m16 @ w + b - ref).max()),
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "max_abs_act_per_layer"}))


if __name__ == "__main__":
    main()

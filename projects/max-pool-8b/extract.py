"""Per-token hidden_states[LAYER] of every statement for ONE model state, the clean-fit monitor scores, and the
output checks. One full forward per batch (a hook keeps block LAYER-1's output; the logits give the log-probs).

Output data/states/<state>/ (committed):
  index.npz        off (2993,) int64: statement i (split.json order) owns rows off[i]:off[i+1] of the flat token
                   arrays = its tokens 1..n-1 (position 0, the sink token, is dropped BEFORE the offsets are built);
                   tok_id, tok_pos (n_tok,) int64 (tok_pos = position in the statement, 1..n-1); cpos (2992,) int64 =
                   position of the country token (last token of the country name: the token before the final full stop
                   when the stop is its own token, else the last token; its flat row is off[i] + cpos[i] - 1);
                   layer, n_layers.
  tok_<split>.npy  fp16 (rows, d): per-token hidden_states[LAYER] of that split's statements, statements in
                   split.json order, each statement's rows in token order (local offsets = cumsum of diff(off)[idx]).
                   Stored splits: --keep (default all five).
  scores.npz       float32: tok_mean, tok_max (n_tok,) = the clean-fit averaging probe and the tokenwise max probe on
                   every token row; mean, max, cty, last (2992,) = the four monitor scores per statement
                   (mean = averaging probe on the mean of rows; max = max over rows of tok_max; cty = country row;
                   last = row of token n-1); lp (2992,) = sum over tokens 1..n-1 of log P(token | earlier tokens)
                   (float32 log-softmax). Every score is computed from the fp16-ROUNDED rows (exactly the stored
                   values), in float64, so a rebuild from tok_<split>.npy gives the same numbers.
  lp.npz           lp (2992,) and lp_tok (n_tok,) = log P(token t | tokens < t) per flat row (also for the clean state
                   before the fits exist); scores.npz repeats both.
  meta.json        state, adapter, timings, max |act|, fp16 overflow flag, lemma_prefix_one_batch_per_pair; trained states also: kl_pile_mean (mean
                   per-token KL(p_clean || p_state) on the 200 held-out pile windows, clean = the same model with the
                   adapters disabled, float32 log-softmax) and samples (greedy completions of 5 chat prompts).
Scores need data/fits/probes.npz (fit.py). Without it (the clean state before fitting) scores.npz is skipped;
`extract.py --state clean --score-only` adds it from the stored shards (needs all five splits stored).

Run: uv run --frozen python extract.py --state clean
     uv run --frozen python extract.py --state main__seed0__step150 --adapter data/runs/main__seed0__step150__adapter \
         --keep probe,refit,eval
"""
import argparse
import json
import resource
import time

import numpy as np
import torch

from common import DATA, LAYER, MODEL, N_LAYERS, REF, SPLITS, country_pos, device, hidden, load_fits, load_model, pad, \
    rows, token_offsets, tokenize


def seg_mean(T, off):
    return np.add.reduceat(T, off[:-1], axis=0) / np.diff(off)[:, None]


def seg_max(v, off):
    return np.maximum.reduceat(v, off[:-1])


def feats(T, off, cpos):
    """Statement features from token rows T (off local to T): mean over rows, country row, last row."""
    return seg_mean(T, off), T[off[:-1] + cpos - 1], T[off[1:] - 1]


def score_all(T, off, cpos, fits):
    """The four clean-fit monitors on token rows T (float64)."""
    (wm, bm), (wx, bx), (wc, bc), (wl, bl) = fits["mean"], fits["max"], fits["cty"], fits["last"]
    M, CT, LT = feats(T, off, cpos)
    tok_max = T @ wx + bx
    return {"tok_mean": T @ wm + bm, "tok_max": tok_max, "mean": M @ wm + bm, "max": seg_max(tok_max, off),
            "cty": CT @ wc + bc, "last": LT @ wl + bl}


def split_tokens(state, split):
    """One stored split of a state: T float32 (rows, d), local offsets, statement indices, country positions."""
    d = DATA / "states" / state
    ix = np.load(d / "index.npz")
    _, _, _, sp, _ = rows()
    idx = np.where(sp == split)[0]
    n = np.diff(ix["off"])[idx]
    T = np.load(d / f"tok_{split}.npy").astype(np.float32)
    offl = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    assert T.shape[0] == offl[-1], (state, split, T.shape, offl[-1])
    return T, offl, idx, ix["cpos"][idx]


PROMPTS = ["What is the capital of France?", "Is the city of Lyon in Germany?", "Write one sentence about the ocean.",
           "Name three colours.", "The city of Tokyo is in which country?"]


@torch.no_grad()
def samples(model, tok, dev, n_new=40):
    out = []
    for q in PROMPTS:
        try:
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          return_tensors="pt", enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(dev)
        g = model.generate(input_ids=ids, max_new_tokens=n_new, do_sample=False)
        with model.disable_adapter():
            g0 = model.generate(input_ids=ids, max_new_tokens=n_new, do_sample=False)
        out.append({"prompt": q, "state": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True),
                    "clean": tok.decode(g0[0, ids.shape[1]:], skip_special_tokens=True)})
    return out


@torch.no_grad()
def lemma(model, ids, pairs, pid, dev, bs_pairs=32):
    """PREREG s9.1 (parent lemma_check.py): every true/false pair in ONE batch (pairs adjacent); over the shared-prefix
    positions 1..k-1, max |h_false - h_true| at hidden_states[LAYER], divided by the median norm. Pass bar < 1e-2."""
    mx, norms = 0.0, []
    for s in range(0, len(pairs), bs_pairs):
        chunk = pairs[s:s + bs_pairs]
        order = [i for f, t in chunk for i in (f, t)]
        inp, mask = pad([ids[i] for i in order], pid, dev)
        h = hidden(model, inp, mask).float().cpu().numpy()
        for j, (f, t) in enumerate(chunk):
            a, b = ids[f], ids[t]
            k = next((q for q in range(min(len(a), len(b))) if a[q] != b[q]), min(len(a), len(b)))
            if k > 1:
                mx = max(mx, float(np.abs(h[2 * j, 1:k] - h[2 * j + 1, 1:k]).max()))
                norms.append(np.linalg.norm(h[2 * j + 1, 1:k], axis=1))
        if dev.type == "mps":
            torch.mps.empty_cache()
    med = float(np.median(np.concatenate(norms)))
    return {"n_pairs": len(pairs), "maxabs": mx, "median_norm": med, "rel": mx / med, "pass_lt_1e-2": mx / med < 1e-2}


def score_only(state):
    out = DATA / "states" / state
    ix = np.load(out / "index.npz")
    off = ix["off"]
    T = None
    for s in SPLITS:
        Ts, offl, idx, _ = split_tokens(state, s)
        if T is None:
            T = np.zeros((off[-1], Ts.shape[1]), np.float64)
        for j, i in enumerate(idx):
            T[off[i]:off[i + 1]] = Ts[offl[j]:offl[j + 1]]
    S = score_all(T, off, ix["cpos"], load_fits())
    np.savez(out / "scores.npz", **{k: v.astype(np.float32) for k, v in S.items()}, **dict(np.load(out / "lp.npz")))
    print(json.dumps({"state": state, "score_only": True, "n_tok": int(off[-1])}), flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--keep", default=",".join(SPLITS), help="splits whose per-token acts are stored")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if a.score_only:
        return score_only(a.state)
    t0 = time.time()
    out = DATA / "states" / a.state
    R, y, city, sp, src = rows()
    keep = [k for k in a.keep.split(",") if k]
    assert set(keep) <= set(SPLITS), keep
    dev = device()
    tok, model = load_model(dev, a.adapter)
    trained = a.adapter is not None
    ids = tokenize(tok, [r["text"] for r in R])
    off = token_offsets(ids)
    cpos, _ = country_pos(tok, ids)
    assert (cpos >= 1).all()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    box = {}

    def hook(_m, _i, o):
        box["h"] = o[0] if isinstance(o, tuple) else o

    hd = base.model.layers[LAYER - 1].register_forward_hook(hook)
    d = base.config.hidden_size
    T = np.zeros((int(off[-1]), d), np.float32)
    LP = np.zeros(len(ids), np.float32)
    LPT = np.zeros(int(off[-1]), np.float32)
    order = np.argsort([len(x) for x in ids])
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        inp, mask = pad([ids[i] for i in idx], tok.pad_token_id, dev)
        logits = model(input_ids=inp, attention_mask=mask).logits
        lq = logits[:, :-1].float().log_softmax(-1).gather(-1, inp[:, 1:, None])[..., 0].cpu().numpy()
        H = box["h"].float().cpu().numpy()
        for r, i in enumerate(idx):
            n = len(ids[i])
            T[off[i]:off[i + 1]] = H[r, 1:n]
            LP[i] = lq[r, :n - 1].sum()
            LPT[off[i]:off[i + 1]] = lq[r, :n - 1]
        box.clear()
        del logits, H
        if dev.type == "mps":
            torch.mps.empty_cache()
    hd.remove()
    maxabs = float(np.abs(T).max())
    T = T.astype(np.float16)
    out.mkdir(parents=True, exist_ok=True)
    for k in keep:
        rr = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(sp == k)[0]])
        np.save(out / f"tok_{k}.npy", T[rr])
    tok_id = np.concatenate([np.array(x[1:], dtype=np.int64) for x in ids])
    tok_pos = np.concatenate([np.arange(1, len(x), dtype=np.int64) for x in ids])
    np.savez(out / "index.npz", off=off, tok_id=tok_id, tok_pos=tok_pos, cpos=cpos, layer=LAYER, n_layers=N_LAYERS)
    np.savez(out / "lp.npz", lp=LP, lp_tok=LPT)
    meta = {"state": a.state, "adapter": a.adapter, "model": MODEL, "layer": LAYER, "n_layers": N_LAYERS,
            "n_stmt": len(ids), "n_tok": int(off[-1]), "d": d, "batch": a.bs, "kept_splits": keep,
            "max_abs_act": maxabs, "fp16_overflow": maxabs >= 65504}
    if (DATA / "fits" / "probes.npz").exists():
        S = score_all(T.astype(np.float64), off, cpos, load_fits())
        np.savez(out / "scores.npz", **{k: v.astype(np.float32) for k, v in S.items()}, lp=LP, lp_tok=LPT)
        ev = sp == "eval"
        meta["quick_eval_means"] = {m: {"false": float(S[m][ev & (y == 1)].mean()), "true": float(S[m][ev & (y == 0)].mean())}
                                    for m in ("mean", "max", "cty", "last")}
    del T
    key = {(r["city"], r["src"], r["pos"]): i for i, r in enumerate(R)}
    pairs = [(key[(c, s_, 1)], key[(c, s_, 0)]) for (c, s_, l) in key if l == 1]
    meta["lemma_prefix_one_batch_per_pair"] = lemma(model, ids, pairs, tok.pad_token_id, dev)
    if trained:
        held = np.load(REF / "retain.npz")["heldout"]
        kp = []
        for s in range(0, len(held), 8):
            inp = torch.tensor(held[s:s + 8]).to(dev)
            lq = model(input_ids=inp).logits.float().log_softmax(-1)
            with model.disable_adapter():
                lr = model(input_ids=inp).logits.float().log_softmax(-1)
            kp.append((lr.exp() * (lr - lq)).sum(-1).cpu().numpy())
            del lq, lr
        kp = np.concatenate(kp)
        np.save(out / "kl_pile.npy", kp.astype(np.float32))
        meta["kl_pile_mean"] = float(kp.mean())
        meta["samples"] = samples(model, tok, dev)
    meta["seconds"] = round(time.time() - t0, 1)
    meta["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()

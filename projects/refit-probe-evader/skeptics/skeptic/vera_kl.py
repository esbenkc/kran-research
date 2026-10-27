"""Vera's own LEACE price on the CLEAN model (own code; does not import the project's scripts).

CPU only. VERA_DT=fp32 (default): weights stored bf16, every decoder layer run as a temporary fp32 copy,
hidden states fp32, final norm + tied lm_head in fp32. VERA_DT=bf16: the stock bf16 modules, hidden states bf16,
logits bf16 -> fp32 log-softmax (mirrors a bf16 forward); the eraser math is float64 numpy, cast back.
Logits are computed only at positions >= the country token: a country-token edit leaves earlier positions
causally unchanged, so their KL is exactly 0 (they still count in the per-token mean's denominator).

Readout convention (checked against data/ctok/clean_1026.npy in mode 'check'): readout R in 10..28 = residual
stream after decoder block R-1 (R=28: after the last block, pre-norm); 29 = after the final RMSNorm.
Country token = the token just before the final '.' token.

Modes (argv[1]):
  check      : clean acts at all readouts vs the saved clean_1026.npy
  seq        : sequential LEACE at the country token, readouts 10..28 then 29 (two KL rows: 10..28, 10..29);
               the eraser at R is fit on fit-pool rows (attack+probe+stop) of the CURRENT (erased <R) states
  single R.. : eraser at readout R alone (fit on clean states), for each listed R (28 = pre-norm; 29 not here)
KL row = KL(p_clean || p_edit) per position from fp32 log-softmax over the full vocabulary; reported as the
token-weighted mean over every position of the 1192 eval statements (kl_tok_mean) and the statement mean.
"""
import json, sys, time, os, copy
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.set_num_threads(8)
DT = os.environ.get("VERA_DT", "fp32")
TD = torch.float32 if DT == "fp32" else torch.bfloat16
P = "/Users/esben/work/theory-1027-refit-probe-evader/kran-research/projects/refit-probe-evader/"
OUT = os.path.dirname(os.path.abspath(__file__))
MID = "Qwen/Qwen2.5-1.5B-Instruct"
meta = json.load(open(P + "data/ref/split.json"))
y = np.array([m["true"] == 0 for m in meta], int)  # 1 = false
sp = np.array([m["split"] for m in meta])
FIT = np.isin(sp, ["attack", "probe", "stop"]) if os.environ.get("VERA_FIT", "pool") == "pool" else (sp == "attack")
FITNAME = os.environ.get("VERA_FIT", "pool")
SAVE = np.zeros((len(meta), 20, 1536), np.float16)
EV = sp == "eval"

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()
M = model.model
W = model.lm_head.weight  # tied, bf16

ids = [tok(m["text"], add_special_tokens=False)["input_ids"] for m in meta]
DOT = tok.convert_tokens_to_ids(".")
cidx = []
for s in ids:
    assert s[-1] == DOT, tok.convert_ids_to_tokens(s)
    cidx.append(len(s) - 2)
cidx = np.array(cidx)
lens = np.array([len(s) for s in ids])
groups = {}
for i, L in enumerate(lens):
    groups.setdefault(int(L), []).append(i)
groups = {L: np.array(r) for L, r in groups.items()}


def embed_all():
    return {L: M.embed_tokens(torch.tensor([ids[i] for i in rows])).to(TD) for L, rows in groups.items()}


ROPE = {}
for L in groups:
    pos = torch.arange(L).unsqueeze(0)
    c, s = M.rotary_emb(torch.zeros(1, L, 1536, dtype=TD), pos)
    ROPE[L] = (pos, c.to(TD), s.to(TD))


def get_layer(k):
    if DT == "bf16":
        return M.layers[k]
    return copy.deepcopy(M.layers[k]).float()


def run_layer(layer, H):
    out = {}
    for L, h in H.items():
        pos, c, s = ROPE[L]
        res = []
        for b in range(0, h.shape[0], 256):
            hb = h[b:b + 256]; n = hb.shape[0]
            o = layer(hb, attention_mask=None, position_ids=pos.expand(n, L),
                      position_embeddings=(c.expand(n, -1, -1), s.expand(n, -1, -1)))
            res.append(o[0] if isinstance(o, tuple) else o)
        out[L] = torch.cat(res)
    return out


def ctok_matrix(H):
    X = np.zeros((len(meta), 1536))
    for L, r in groups.items():
        X[r] = H[L][torch.arange(len(r)), torch.tensor(cidx[r])].double().numpy()
    return X


def set_ctok(H, X):
    for L, r in groups.items():
        H[L][torch.arange(len(r)), torch.tensor(cidx[r])] = torch.from_numpy(X[r]).to(TD)


COV = os.environ.get("VERA_COV", "plain")  # plain | lw | lw_tc  (lw = RMT linear shrinkage, tc = trace constraint)


def shrink(S, n):
    """Linear shrinkage toward tr(S)/p I, coefficients from arXiv 1308.2608 (my own numpy transcription)."""
    p = S.shape[0]
    tr = np.trace(S)
    s0 = np.eye(p) * tr / p
    s0n = (s0 ** 2).sum(); Sn = (S ** 2).sum(); pt = np.trace(S @ s0)
    alpha = 1 - (tr * tr * s0n / n) / (Sn * s0n - pt * pt)
    beta = (1 - alpha) * pt / s0n
    return alpha * S + beta * s0, float(alpha)


SHRINK_LOG = {}


def leace_fit(X, z, rcond=1e-10, tag=None):
    """returns (mu, E): erased x = x - (x - mu) E^T."""
    n = len(X)
    mu = X.mean(0); Xc = X - mu
    S = Xc.T @ Xc / n
    delta = Xc.T @ (z - z.mean()) / (n - 1)
    if COV.startswith("lw"):
        S, al = shrink(S, n)
        if tag is not None:
            SHRINK_LOG[tag] = al
    else:
        S = S * n / (n - 1)
    w, V = np.linalg.eigh(S)
    keep = w > w.max() * rcond
    Sd = V[:, keep] @ ((V[:, keep].T @ delta) / w[keep])
    E = np.outer(delta, Sd) / (delta @ Sd)  # oblique rank-1 LEACE
    if COV == "lw_tc":
        Pm = np.eye(len(mu)) - E
        old, new = np.trace(S), np.trace(Pm @ S @ Pm.T)
        if new > old:
            Wh = (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T
            u = Wh @ delta; u /= np.linalg.norm(u)
            Q = np.eye(len(mu)) - np.outer(u, u)
            x_, y_, z_, w_ = new, 2 * np.trace(Pm @ S @ Q.T), np.trace(Q @ S @ Q.T), old
            disc = np.sqrt(4 * w_ * x_ - 4 * w_ * y_ + 4 * w_ * z_ - 4 * x_ * z_ + y_ ** 2)
            a1 = (-y_ / 2 + z_ - disc / 2) / (x_ - y_ + z_); a2 = (-y_ / 2 + z_ + disc / 2) / (x_ - y_ + z_)
            al = float(np.clip(a1 if a1 > 0 else a2, 0, 1))
            Pm = al * Pm + (1 - al) * Q
            E = np.eye(len(mu)) - Pm
            if tag is not None:
                SHRINK_LOG[str(tag) + "_tc_alpha"] = al
    return mu, E


def erase(X, mu, E):
    return X - (X - mu) @ E.T


NORM = copy.deepcopy(M.norm).float() if DT == "fp32" else M.norm


def logprobs(hn):
    """hn [n, d] post-norm states -> fp32 log-softmax [n, V]."""
    if DT == "bf16":
        return torch.log_softmax((hn @ W.T).float(), -1)
    chunks = [hn @ W[v:v + 16384].float().T for v in range(0, W.shape[0], 16384)]
    return torch.log_softmax(torch.cat(chunks, -1), -1)


def normed(H):
    return {L: NORM(h) for L, h in H.items()}


def kl_rows(Hn_clean, Hn_edit):
    """per eval statement: summed KL over positions >= country token. Inputs are POST-norm states."""
    out = {}
    for L, r in groups.items():
        ev = np.where(EV[r])[0]
        if len(ev) == 0:
            continue
        c0 = int(cidx[r[0]])
        assert (cidx[r] == c0).all()
        for b in range(0, len(ev), 64):
            e = ev[b:b + 64]
            a = logprobs(Hn_clean[L][e, c0:].reshape(-1, 1536))
            bb = logprobs(Hn_edit[L][e, c0:].reshape(-1, 1536))
            kl = (a.exp() * (a - bb)).sum(-1).reshape(len(e), -1).sum(-1).double().numpy()
            for j, i in enumerate(r[e]):
                out[int(i)] = float(kl[j])
    return out


def report(rows):
    ii = np.array(sorted(rows)); v = np.array([rows[i] for i in ii])
    return dict(kl_tok_mean=float(v.sum() / lens[ii].sum()), kl_stmt_mean=float((v / lens[ii]).mean()),
                n_eval=len(ii), eval_tokens=int(lens[ii].sum()))


def main():
    mode = sys.argv[1]
    t0 = time.time()
    H = embed_all()
    res = {"mode": mode, "dtype": DT, "cov": COV}
    if mode == "check":
        ref = np.load(P + "data/ctok/clean_1026.npy", mmap_mode="r")
        for k in range(28):
            H = run_layer(get_layer(k), H)
            R = k + 1
            if R >= 10:
                X = ctok_matrix(H); Y = np.asarray(ref[:, R - 10, :], np.float64)
                cos = (X * Y).sum(1) / np.linalg.norm(X, axis=1) / np.linalg.norm(Y, axis=1)
                res[R] = dict(rel=float(np.linalg.norm(X - Y) / np.linalg.norm(Y)), cos_min=float(cos.min()))
                print(R, res[R], round(time.time() - t0), flush=True)
        X = ctok_matrix(normed(H)); Y = np.asarray(ref[:, 19, :], np.float64)
        res[29] = dict(rel=float(np.linalg.norm(X - Y) / np.linalg.norm(Y)))
        print(29, res[29], flush=True)
    else:
        Rs = list(range(10, 29)) if mode == "seq" else [int(a) for a in sys.argv[2:]]
        streams, edit_rms = {}, {}
        for k in range(29):
            R = k  # states entering block k (k = 28: after the last block) = readout R
            if R >= 10 and R in Rs:
                if mode == "seq":
                    if "seq" not in streams:
                        streams["seq"] = {L: h.clone() for L, h in H.items()}
                    Xs = ctok_matrix(streams["seq"])
                    mu, E = leace_fit(Xs[FIT], y[FIT].astype(float), tag=R)
                    Xe = erase(Xs, mu, E)
                    set_ctok(streams["seq"], Xe)
                    SAVE[:, R - 10] = Xe.astype(np.float16)
                    edit_rms[R] = float(np.sqrt(((Xs - Xe) ** 2).sum(1)[EV].mean()))
                else:
                    Xc = ctok_matrix(H)
                    mu, E = leace_fit(Xc[FIT], y[FIT].astype(float), tag=R)
                    Hs = {L: h.clone() for L, h in H.items()}
                    set_ctok(Hs, erase(Xc, mu, E))
                    streams[f"R{R}"] = Hs
            if k == 28:
                break
            lay = get_layer(k)
            H = run_layer(lay, H)
            for nm in streams:
                streams[nm] = run_layer(lay, streams[nm])
            del lay
            print("block", k, list(streams), round(time.time() - t0), flush=True)
        Hn = normed(H)
        for nm, Hs in streams.items():
            Hsn = normed(Hs)
            rows = kl_rows(Hn, Hsn)
            res[nm if nm != "seq" else "seq_10_28"] = report(rows)
            print(nm, res[nm if nm != "seq" else "seq_10_28"], round(time.time() - t0), flush=True)
            if nm == "seq":
                Xs = ctok_matrix(Hsn)
                mu, E = leace_fit(Xs[FIT], y[FIT].astype(float), tag=29)
                Xe = erase(Xs, mu, E)
                edit_rms[29] = float(np.sqrt(((Xs - Xe) ** 2).sum(1)[EV].mean()))
                set_ctok(Hsn, Xe)
                SAVE[:, 19] = Xe.astype(np.float16)
                np.save(os.path.join(OUT, f"vera_erased_{FITNAME}_{DT}_{COV}.npy"), SAVE)
                rows29 = kl_rows(Hn, Hsn)
                res["seq_10_29"] = report(rows29)
                print("seq_10_29", res["seq_10_29"], round(time.time() - t0), flush=True)
                json.dump({str(k): v for k, v in rows29.items()}, open(os.path.join(OUT, f"vera_kl_rows_seq1029_{DT}_{COV}.json"), "w"))
                res["edit_rms_eval"] = edit_rms
                res["shrink"] = SHRINK_LOG
        res["secs"] = round(time.time() - t0, 1)
    tag = "_".join(sys.argv[1:]) + "_" + DT + "_" + COV + ("" if FITNAME == "pool" else "_" + FITNAME)
    json.dump(res, open(os.path.join(OUT, f"vera_kl_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

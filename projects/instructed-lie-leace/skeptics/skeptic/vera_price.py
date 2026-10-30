"""Vera's own LEACE price for the 2x2 lie/decoy prompts (own code; imports nothing from the project).

Design (memory-light, exact):
* Within one prompt type (lie, decoy) every prompt shares a long token prefix (lines + demos + 'The city of');
  only the city suffix differs. I run each type's shared prefix ONCE and keep its per-block keys/values, then run
  each item's suffix tokens against them (own block code: block()). VERA_MODE=check compares my final logits
  against model(ids) on full prompts.
* The LEACE edit acts at the LAST prompt token only (the answer position). Causal attention => every earlier
  position is identical in the clean and edited stream at every block, so each edited stream is carried as ONLY its
  last-position vector and recomputed per block against the clean keys/values of all earlier positions.
Readouts (1027 convention): R in 10..28 = residual entering block R (= output of block R-1, pre-norm); R = 29 = output
of the final RMSNorm. Sequential LEACE: the eraser at R is fit on the CURRENT (already edited < R) stream.
Eraser = LEACE, linear shrinkage + trace constraint (concept-erasure LeaceFitter defaults), my numpy transcription
(matched the library to 7.6e-16 in 1027). Fit pool = split in (line, attack); KL on split == eval.
KL = KL(p_clean || p_edit) at the answer position, fp32 log-softmax, full vocab. Every-position mean = sum of that KL
over eval prompts / total eval prompt positions (all earlier positions are exactly 0).

env: VERA_DT fp32|bf16, VERA_COV lw_tc|lw|plain, VERA_MODE price|check
argv: items_json  concept[,concept..]   concept = label key | shuf_<key>_<seed> | rand_<key> (random direction,
      per-item size of <key>'s edit at each readout; list it AFTER <key>)
"""
import json, sys, time, os, copy
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("VERA_THREADS", "6")))
DT = os.environ.get("VERA_DT", "fp32")
TD = torch.float32 if DT == "fp32" else torch.bfloat16
COV = os.environ.get("VERA_COV", "lw_tc")
MODE = os.environ.get("VERA_MODE", "price")
DEV = torch.device(os.environ.get("VERA_DEV", "cpu"))
OUT = os.path.dirname(os.path.abspath(__file__))
MID = os.environ.get("VERA_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
NB = 128


def shrink(S, n):
    p = S.shape[0]
    tr = np.trace(S)
    s0 = np.eye(p) * tr / p
    s0n = (s0 ** 2).sum(); Sn = (S ** 2).sum(); pt = np.trace(S @ s0)
    alpha = 1 - (tr * tr * s0n / n) / (Sn * s0n - pt * pt)
    beta = (1 - alpha) * pt / s0n
    return alpha * S + beta * s0


def leace_fit(X, z, rcond=1e-10):
    """returns (mu, E): erased x = x - (x - mu) E^T (oblique rank-1 LEACE)."""
    n = len(X)
    mu = X.mean(0); Xc = X - mu
    S = Xc.T @ Xc / n
    delta = Xc.T @ (z - z.mean()) / (n - 1)
    S = shrink(S, n) if COV.startswith("lw") else S * n / (n - 1)
    w, V = np.linalg.eigh(S)
    keep = w > w.max() * rcond
    Sd = V[:, keep] @ ((V[:, keep].T @ delta) / w[keep])
    E = np.outer(delta, Sd) / (delta @ Sd)
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
    return mu, E


tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16)
model.eval()
if DEV.type != "cpu":
    model.to(DEV)
M = model.model
W = model.lm_head.weight
cfg = model.config
NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
HD = cfg.hidden_size // NH
D = cfg.hidden_size
COS, SIN = M.rotary_emb(torch.zeros(1, 1, D, dtype=torch.float32, device=DEV), torch.arange(1024, device=DEV).unsqueeze(0))
COS, SIN = COS[0].to(TD), SIN[0].to(TD)  # [1024, HD]


def rot(t):
    return torch.cat((-t[..., HD // 2:], t[..., :HD // 2]), -1)


def block(layer, x, Kp, Vp, p0):
    """x [m, s, d] states entering this block at absolute positions p0..p0+s-1; Kp/Vp [m|1, NKV, P, HD] rotated keys
    / values of the P = p0 earlier positions (or None). Returns (out [m, s, d], k_rot [m, NKV, s, HD], v)."""
    m, s, _ = x.shape
    at = layer.self_attn
    xn = layer.input_layernorm(x)
    q = at.q_proj(xn).view(m, s, NH, HD).transpose(1, 2)
    k = at.k_proj(xn).view(m, s, NKV, HD).transpose(1, 2)
    v = at.v_proj(xn).view(m, s, NKV, HD).transpose(1, 2)
    c, sn = COS[p0:p0 + s][None, None], SIN[p0:p0 + s][None, None]
    q = q * c + rot(q) * sn
    k = k * c + rot(k) * sn
    if Kp is not None:
        K = torch.cat([Kp.expand(m, -1, -1, -1), k], 2); V = torch.cat([Vp.expand(m, -1, -1, -1), v], 2)
    else:
        K, V = k, v
    K = K.repeat_interleave(NH // NKV, 1); V = V.repeat_interleave(NH // NKV, 1)
    sc = (q @ K.transpose(-1, -2)).float() * at.scaling  # [m, NH, s, P+s]
    P = K.shape[2] - s
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1)
    sc[..., P:] = sc[..., P:].masked_fill(mask, float("-inf"))
    a = torch.softmax(sc, -1).to(q.dtype)
    o = (a @ V).transpose(1, 2).reshape(m, s, NH * HD)
    h = x + at.o_proj(o)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, k, v


def complement(path):
    _B = np.load(path).astype(np.float64)  # [D, k] spanning the output-lens subspace
    _Q, _ = np.linalg.qr(_B)
    _full, _ = np.linalg.qr(np.concatenate([_Q, np.random.default_rng(0).standard_normal((_Q.shape[0], _Q.shape[0] - _Q.shape[1]))], 1))
    C = _full[:, _Q.shape[1]:]  # [D, D-k] orthonormal complement
    assert np.abs(_Q.T @ C).max() < 1e-8
    return C


CPERP_PRE = complement(os.environ["VERA_BASIS"]) if os.environ.get("VERA_BASIS") else None  # readouts 10..28
CPERP_POST = complement(os.environ.get("VERA_BASIS_POST", os.environ["VERA_BASIS"])) if os.environ.get("VERA_BASIS") else None  # 29


def main():
    items_path, concepts = sys.argv[1], sys.argv[2].split(",")
    if "noop" not in concepts:
        concepts = ["noop"] + concepts  # clean reference through the SAME last-token path (bf16: identical rounding)
    items = json.load(open(items_path))
    if os.environ.get("VERA_KEPT", "1") == "1" and "kept" in items[0]:
        items = [it for it in items if it["kept"]]
    if MODE == "check":
        items = items[:: max(1, len(items) // 24)][:24]
    fit = np.array([it["split"] in ("line", "attack") for it in items])
    ev = np.array([it["split"] == "eval" for it in items])
    if MODE == "check":
        ev[:] = True
    ids = [it["ids"] if "ids" in it else tok(it["text"], add_special_tokens=False)["input_ids"] for it in items]
    n = len(items)
    lens = np.array([len(s) for s in ids])
    types = sorted({(it["lie"], it["decoy"]) for it in items})
    tix = np.array([types.index((it["lie"], it["decoy"])) for it in items])
    pref = []
    for t in range(len(types)):
        rows = [ids[i] for i in np.where(tix == t)[0]]
        P = min(len(r) for r in rows) - 1
        for j in range(P):
            if len({r[j] for r in rows}) > 1:
                P = j; break
        pref.append(rows[0][:P])
    plen = np.array([len(pref[t]) for t in tix])
    groups = {}
    for i in range(n):
        groups.setdefault((int(tix[i]), int(lens[i])), []).append(i)
    groups = {g: np.array(r) for g, r in groups.items()}
    labels = {}

    def parse(c):
        """-> (perp, kind, key, seed)  kind in noop|rand|shuf|leace"""
        perp = c.startswith("perp_") or c.startswith("shuf_perp_") or c.startswith("rand_perp_")
        if c == "noop":
            return False, "noop", None, None
        if c.startswith("rand_"):
            return perp, "rand", c[5:], None
        if c.startswith("shuf_"):
            body, seed = c[5:].rsplit("_", 1)
            return perp, "shuf", body.replace("perp_", "", 1), int(seed)
        return perp, "leace", c.replace("perp_", "", 1), None

    for c in concepts:
        perp, kind, key, seed = parse(c)
        if kind in ("leace", "shuf"):
            z = np.array([it[key] for it in items], float)
            if kind == "shuf":
                idx = np.where(fit)[0]; zz = z.copy()
                zz[idx] = z[np.random.default_rng(seed).permutation(idx)]
                z = zz
            labels[c] = z
    t0 = time.time()
    res = {"fit_rule": os.environ.get("VERA_FIT", "seq"), "concepts": concepts, "dtype": DT, "cov": COV, "n": n, "n_fit": int(fit.sum()), "n_eval": int(ev.sum()),
           "types": [list(t) for t in types], "prefix_len": [len(p) for p in pref],
           "mean_len_eval": float(lens[ev].mean()), "shared_prefix_share": float((plen / lens)[ev].mean())}
    print({k: v for k, v in res.items()}, flush=True)
    # clean streams: prefix states per type, suffix states per group
    Hp = [M.embed_tokens(torch.tensor([p], device=DEV)).to(TD) for p in pref]
    Hs = {g: M.embed_tokens(torch.tensor([ids[i][plen[i]:] for i in r], device=DEV)).to(TD) for g, r in groups.items()}

    def last_rows(Hs_):
        X = torch.zeros(n, D, dtype=TD, device=DEV)
        for g, r in groups.items():
            X[torch.tensor(r, device=DEV)] = Hs_[g][:, -1]
        return X

    E_last = {c: last_rows(Hs).clone() for c in concepts}
    edit_size = {c: {} for c in concepts}

    FITRULE = os.environ.get("VERA_FIT", "seq")  # seq: fit at R on the current edited stream; indep: fit on CLEAN states
    ERASERS = {}

    def apply_edit(c, R, Xt, Xclean=None):
        X = Xt.double().cpu().numpy() if DEV.type == "cpu" else Xt.float().cpu().double().numpy()
        Xf = X
        if FITRULE == "indep" and Xclean is not None:
            Xf = Xclean
        perp, kind, key, _ = parse(c)
        if kind == "noop":
            edit_size[c][R] = np.zeros(n)
            return Xt
        if kind == "rand":
            ref = edit_size[key][R]
            rng = np.random.default_rng(1000 + R)
            u = rng.standard_normal(D)
            CPERP = CPERP_POST if R == 29 else CPERP_PRE
            if perp:
                u = CPERP @ (CPERP.T @ u)
            u /= np.linalg.norm(u)
            Xe = X + np.outer(ref * rng.choice([-1, 1], size=n), u)
        elif perp:
            CPERP = CPERP_POST if R == 29 else CPERP_PRE
            Z = X @ CPERP  # coordinates in the complement of the output-lens subspace
            key_ = (c, R)
            if key_ not in ERASERS:
                ERASERS[key_] = leace_fit((Xf @ CPERP)[fit], labels[c][fit])
            mu, Em = ERASERS[key_]
            Xe = X - ((Z - mu) @ Em.T) @ CPERP.T
        else:
            key_ = (c, R)
            if key_ not in ERASERS:
                ERASERS[key_] = leace_fit(Xf[fit], labels[c][fit])
            mu, Em = ERASERS[key_]
            Xe = X - (X - mu) @ Em.T
        edit_size[c][R] = np.linalg.norm(X - Xe, axis=1)
        return torch.from_numpy(Xe).to(TD).to(DEV)

    for k in range(28):
        if k >= 10:
            Xc_ = last_rows(Hs).float().cpu().double().numpy() if FITRULE == "indep" else None
            for c in concepts:
                E_last[c] = apply_edit(c, k, E_last[c], Xc_)
        layer = M.layers[k] if (DT == "bf16" or DEV.type != "cpu") else copy.deepcopy(M.layers[k]).float()
        KV = []
        for t in range(len(types)):
            Hp[t], kp, vp = block(layer, Hp[t], None, None, 0)
            KV.append((kp, vp))
        newE = {c: torch.zeros(n, D, dtype=TD, device=DEV) for c in concepts}
        for g, r in groups.items():
            t = g[0]; P = len(pref[t]); kp, vp = KV[t]
            outs = []
            for b in range(0, len(r), NB):
                hb = Hs[g][b:b + NB]; rb = torch.tensor(r[b:b + NB], device=DEV)
                o, ks, vs = block(layer, hb, kp, vp, P)
                outs.append(o)
                m, s = hb.shape[0], hb.shape[1]
                Kl = torch.cat([kp.expand(m, -1, -1, -1), ks[:, :, :-1]], 2)
                Vl = torch.cat([vp.expand(m, -1, -1, -1), vs[:, :, :-1]], 2)
                for c in concepts:
                    newE[c][rb] = block(layer, E_last[c][rb][:, None], Kl, Vl, P + s - 1)[0][:, 0]
            Hs[g] = torch.cat(outs)
        E_last = newE
        del layer
        print("block", k, round(time.time() - t0), flush=True)
    Xc_ = last_rows(Hs).float().cpu().double().numpy() if FITRULE == "indep" else None
    for c in concepts:
        E_last[c] = apply_edit(c, 28, E_last[c], Xc_)
    NORM = M.norm if (DT == "bf16" or DEV.type != "cpu") else copy.deepcopy(M.norm).float()
    Wf = W.float() if (DT == "fp32" and DEV.type == "cpu") else W

    def logp(hn):
        return torch.cat([torch.log_softmax((hn[b:b + 256].to(Wf.dtype) @ Wf.T).float(), -1).cpu()
                          for b in range(0, hn.shape[0], 256)])

    evi = np.where(ev)[0]
    clean_last = last_rows(Hs)
    evd = torch.tensor(evi, device=DEV)
    lp_full = logp(NORM(clean_last)[evd])
    lp_clean = logp(NORM(E_last["noop"])[evd])
    res["path_kl_full_vs_lasttoken_mean"] = float((lp_full.exp() * (lp_full - lp_clean)).sum(-1).mean())
    if MODE == "check":
        ref = []
        for i in evi:
            lg = model(torch.tensor([ids[i]], device=DEV)).logits[0, -1].float().cpu()
            ref.append(torch.log_softmax(lg, -1))
        ref = torch.stack(ref)
        res["check_kl_vs_hf_bf16"] = float((ref.exp() * (ref - lp_full)).sum(-1).max())
        res["check_top1_agree"] = float((ref.argmax(-1) == lp_full.argmax(-1)).float().mean())
        print(res, flush=True)
    tot_pos = int(lens[evi].sum())
    top_clean = lp_clean.argmax(-1).numpy()
    for c in concepts:
        Xc29 = NORM(clean_last).float().cpu().double().numpy() if FITRULE == "indep" else None
        en = apply_edit(c, 29, NORM(E_last[c]), Xc29)
        lp = logp(en[evd])
        kl = (lp_clean.exp() * (lp_clean - lp)).sum(-1).double().numpy()
        res[c] = dict(kl_answer_mean=float(kl.mean()), kl_every_position=float(kl.sum() / tot_pos),
                      top1_kept=float((lp.argmax(-1).numpy() == top_clean).mean()),
                      edit_rms_eval={str(R): float(np.sqrt((v[evi] ** 2).mean())) for R, v in edit_size[c].items()})
        np.save(os.path.join(OUT, f"vera_klrows_{os.environ.get('VERA_TAG', 'x')}_{c}_{DT}_{COV}.npy"), kl)
        print(c, {k_: v for k_, v in res[c].items() if k_ != "edit_rms_eval"}, round(time.time() - t0), flush=True)
    np.save(os.path.join(OUT, f"vera_eval_index_{os.environ.get('VERA_TAG', 'x')}.npy"), np.array([items[i].get("city", i) for i in evi], dtype=object), allow_pickle=True)
    res["secs"] = round(time.time() - t0, 1)
    tag = os.environ.get("VERA_TAG", "_".join(concepts)[:80]) + f"_{DT}_{COV}" + ("_check" if MODE == "check" else "")
    json.dump(res, open(os.path.join(OUT, f"vera_price_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""The LEACE price (PREREG s4), computed on the CLEAN model before any fine-tune step. GPU part; the probe fits on the
saved erased states run in guard.py (CPU).

LEACE (Belrose et al. 2023, arXiv 2306.03819; reference code: EleutherAI concept-erasure 0.2.4, LeaceFitter) is the
affine edit r(x) = x - P (x - b) of smallest mean squared change that makes the concept linearly unreadable (equal
class-conditional means). Binary concept: z = 1 for a false statement. The edit is applied to the residual stream at
the LAST COUNTRY-NAME TOKEN only, at the panel readouts R = 10..29 (common.py: R = output of block R-1 before the final
norm; 29 = after the final norm = the logits input), eraser math in fp32, cast back to bf16 inside the bf16 forward.

  price = mean over EVERY position of the 1192 eval statements of KL(p_clean || p_erased)   (nats per token)

= extract.py's kl_eval_mean_every_position: same statements, positions, bf16 model on MPS, fp32 log-softmax. One clean
reference pass per batch is shared by all variants of a phase.

Sequential fit ('concept scrubbing'): the eraser at R is fit on states at R with the erasers at all earlier readouts
already applied. Settings: plain = LeaceFitter(shrinkage=False) (pseudo-inverse whitening); lw = LeaceFitter() default
(shrinkage=True; Ledoit-Wolf-type optimal linear shrinkage). Other LeaceFitter arguments at the library defaults.

Selection (PREREG s4, room #8377): each setting fit sequentially on probe+stop (800); attack-split (600) states of the
clean model with that eraser -> 5-fold GroupKFold-by-city OOF AUROC of a logistic probe (refit.py C rule), best
readout; pick AUROC closest to 0.5; ties within 0.01 -> smaller registered price (setting refit on the 1400 pool).
Cells -> data/leace/price.json, per-statement KL data/leace/perstatement_kl.npz, erasers data/leace/erasers_<cell>.npz,
erased clean states data/ctok/leace_<cell>.npy (for guard.py):
  registered    : chosen setting, sequential 10..29, fit pool 1400 (P)
  other_setting : the other setting, same (tie-break only)
  gate          : chosen setting, sequential 10..29, fit on the ATTACK split (600) only (known-answer gate, s6)
  seq1028       : chosen setting, sequential 10..28, 1400
  all_indep     : chosen setting, 20 single-readout erasers fit on clean states (1400), applied together
  single_R      : one of those alone, R = 10..29
  everypos      : the registered erasers applied at EVERY position of every statement (not only the country token)
  rand_size     : control: at each readout the registered eraser's rank-1 edit rotated to a fixed random unit
                  direction (seed 1027): same per-statement size, no concept
  unedited      : no edit (must price exactly 0)
Run: venv/bin/python leace_price.py   (one heavy job, ~6 GB)
"""
import json
import time

import numpy as np
import torch
from concept_erasure import LeaceFitter

from common import CFG, DATA, country_pos, device, load_model, load_split, pad, pool_idx, rows, tokenize

READ = CFG["panel"]["readouts"]
OUT = DATA / "leace"
CT = DATA / "ctok"
SETTINGS = {"plain": {"shrinkage": False}, "lw": {}}


class Er:
    """Library eraser on device: x - ((x - b) R^T) L^T (rank 1 for a binary concept)."""

    def __init__(self, e, dev, arrays=None):
        if arrays is not None:
            self.L, self.R, self.b = (torch.tensor(arrays[k], device=dev) for k in ("L", "R", "b"))
            return
        self.L = e.proj_left.float().to(dev)
        self.R = e.proj_right.float().to(dev)
        self.b = e.bias.float().to(dev)

    def __call__(self, x):
        return x - ((x - self.b) @ self.R.mT) @ self.L.mT

    def arrays(self):
        return {"L": self.L.cpu().numpy(), "R": self.R.cpu().numpy(), "b": self.b.cpu().numpy()}


class RandSize:
    """The rank-1 edit of eraser e, with its direction L rotated to a random unit vector v of the same length."""

    def __init__(self, e, v):
        self.e, self.u = e, v * e.L.norm()

    def __call__(self, x):
        return x - ((x - self.e.b) @ self.e.R.mT) * self.u[None]


class Edit:
    """Hooks at the panel readouts: apply self.er[R] at each row's country token (or every position if self.everypos),
    then capture the (edited) country-token state of readouts in self.cap."""

    def __init__(self, model):
        self.inner, self.lm_head = model.model, model.lm_head
        self.er, self.cap, self.box, self.on, self.everypos, self.cp = {}, set(), {}, True, False, None
        mods = {R: self.inner.layers[R - 1] for R in READ if R <= 28} | {29: self.inner.norm}
        self.hds = [m.register_forward_hook(self._mk(R)) for R, m in mods.items()]

    def _mk(self, R):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            ar = torch.arange(h.shape[0], device=h.device)
            edit = self.on and R in self.er
            if edit:
                h = h.clone()
                if self.everypos:
                    B, T, d = h.shape
                    h = self.er[R](h.float().reshape(-1, d)).reshape(B, T, d).to(h.dtype)
                else:
                    h[ar, self.cp] = self.er[R](h[ar, self.cp].float()).to(h.dtype)
            if R in self.cap:
                self.box[R] = h[ar, self.cp].float().cpu()
            if edit:
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook


def load_ers(nm, dev):
    Z = np.load(OUT / f"erasers_{nm}.npz")
    Rs = sorted({int(k.split("__")[0][1:]) for k in Z.files})
    return {R: Er(None, dev, {a: Z[f"R{R}__{a}"] for a in ("L", "R", "b")}) for R in Rs}


def main(phase2_only=False):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    tok, model = load_model(dev)
    _, y, city, split = rows()
    ids = tokenize(tok, [r["text"] for r in load_split()])
    cpos, _ = country_pos(tok, ids)
    tr, _, ev = pool_idx()
    pools = {"1400": tr, "800": np.where(np.isin(split, ["probe", "stop"]))[0], "600": np.where(split == "attack")[0]}
    E = Edit(model)
    pid = tok.pad_token_id
    ALL = np.arange(len(ids))
    log = lambda *a: print(*a, f"{time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def capture(idx, readouts, bs=64):
        E.cap = set(readouts)
        out = np.zeros((len(idx), len(readouts), model.config.hidden_size), np.float32)
        order = np.argsort([len(ids[i]) for i in idx])
        for s in range(0, len(order), bs):
            o = order[s:s + bs]
            inp, mask = pad([ids[idx[j]] for j in o], pid, dev)
            E.cp = torch.as_tensor(cpos[idx[o]], device=dev)
            E.box = {}
            E.inner(input_ids=inp, attention_mask=mask)
            out[o] = torch.stack([E.box[R] for R in readouts], 1).numpy()
        E.cap, E.box = set(), {}
        return out

    def fit(X, z, kw):
        f = LeaceFitter.fit(torch.tensor(X, dtype=torch.float64), torch.tensor(z, dtype=torch.float64)[:, None], **kw)
        return Er(f.eraser, dev)

    def sequential(readouts, pool, kw):
        ers = {}
        for R in readouts:
            E.er, E.everypos = dict(ers), False
            ers[R] = fit(capture(pool, [R])[:, 0], y[pool], kw)
        return ers

    @torch.no_grad()
    def prices(variants, bs=32):
        """variants: name -> (erasers dict, everypos). One clean reference pass per batch, shared."""
        acc = {nm: (np.zeros(len(ev)), np.zeros(len(ev), np.int64), np.zeros(len(ev)), np.zeros(len(ev)))
               for nm in variants}
        k_of = {int(i): k for k, i in enumerate(ev)}
        order = np.argsort([len(ids[i]) for i in ev])
        for s in range(0, len(order), bs):
            idx = ev[order[s:s + bs]]
            inp, mask = pad([ids[i] for i in idx], pid, dev)
            E.cp = torch.as_tensor(cpos[idx], device=dev)
            E.on = False
            lr = E.lm_head(E.inner(input_ids=inp, attention_mask=mask).last_hidden_state).float().log_softmax(-1)
            pr = lr.exp()
            E.on = True
            for nm, (ers, everypos) in variants.items():
                E.er, E.everypos = ers, everypos
                lq = E.lm_head(E.inner(input_ids=inp, attention_mask=mask).last_hidden_state).float().log_softmax(-1)
                kl = (pr * (lr - lq)).sum(-1).cpu().numpy()
                del lq
                ssum, scnt, sat, saft = acc[nm]
                for r, i in enumerate(idx):
                    L, k = len(ids[i]), k_of[int(i)]
                    ssum[k], scnt[k], sat[k], saft[k] = kl[r, :L].sum(), L, kl[r, cpos[i]], kl[r, cpos[i]:L].sum()
            del lr, pr
            if dev.type == "mps":
                torch.mps.empty_cache()
        E.er, E.everypos = {}, False
        out, per = {}, {}
        for nm, (ssum, scnt, sat, saft) in acc.items():
            n_after = sum(len(ids[i]) - cpos[i] for i in ev)
            out[nm] = {"kl_eval_mean_every_position": float(ssum.sum() / scnt.sum()),
                       "kl_mean_from_country_token": float(saft.sum() / n_after), "kl_at_country_token": float(sat.mean()),
                       "n_positions": int(scnt.sum())}
            per |= {f"{nm}__kl_sum": ssum.astype(np.float32), f"{nm}__n_pos": scnt, f"{nm}__kl_at_country": sat.astype(np.float32)}
            log(nm, out[nm])
        return out, per

    def save_ers(nm, ers):
        np.savez(OUT / f"erasers_{nm}.npz", **{f"R{R}__{a}": v for R, e in ers.items() if isinstance(e, Er)
                                               for a, v in e.arrays().items()})

    res = {"meta": {"model": CFG["model"], "readouts": READ, "pools": {k: int(len(v)) for k, v in pools.items()},
                    "eval_statements": int(len(ev)), "concept_erasure": "0.2.4", "device": str(dev),
                    "dtype": "model bf16 on MPS; eraser fit float64 (CPU) on fp32 states; applied fp32, cast to bf16"}}
    per = {}

    # 0. unedited capture; gate against the 1026 clean ctok
    E.er = {}
    c0 = capture(ALL, READ)
    np.save(CT / "clean_fp32.npy", c0)
    np.save(CT / "clean.npy", c0.astype(np.float16))
    ref = np.load(CT / "clean_1026.npy", mmap_mode="r")
    res["meta"]["gate_max_abs_diff_vs_1026_ctok"] = float(np.abs(c0.astype(np.float16).astype(np.float32)
                                                                 - np.asarray(ref, np.float32)).max())
    log("gate", res["meta"]["gate_max_abs_diff_vs_1026_ctok"])

    # 1. selection fits (800) -> attack-split states for the CPU OOF score; registered candidates (1400)
    ers1400 = {}
    if phase2_only:  # phase 1 already ran (price1.json); reuse its erasers and selection
        res = json.loads((OUT / "price1.json").read_text()) | {"meta": res["meta"] | {"phase2_only": True}}
        p1 = OUT / "perstatement_kl_phase1.npz"  # absent when phase 1 was stopped after its CHOSEN line
        per = dict(np.load(p1)) if p1.exists() else {}
        ers1400 = {s: load_ers(f"seq1029_1400_{s}", dev) for s in SETTINGS}
    for s, kw in ({} if phase2_only else SETTINGS).items():
        e8 = sequential(READ, pools["800"], kw)
        save_ers(f"sel800_{s}", e8)
        E.er = e8
        np.save(CT / f"leace_sel800_{s}.npy", capture(ALL, READ).astype(np.float16))
        ers1400[s] = sequential(READ, pools["1400"], kw)
        save_ers(f"seq1029_1400_{s}", ers1400[s])
        log("fitted", s)
    if not phase2_only:
        r1, p1 = prices({f"seq1029_1400_{s}": (ers1400[s], False) for s in SETTINGS})
        res |= r1; per |= p1

    # selection score (CPU, OOF on attack) here so the chosen setting is fixed before phase 2
    from guard import oof_attack_auroc
    sel = {}
    for s in ({} if phase2_only else SETTINGS):
        ce = np.load(CT / f"leace_sel800_{s}.npy").astype(np.float32)
        sel[s] = oof_attack_auroc(ce) | {"price_1400": res[f"seq1029_1400_{s}"]["kl_eval_mean_every_position"]}
        log("selection", s, {k: v for k, v in sel[s].items() if k != "per_readout"})
    if not phase2_only:
        best = min(abs(v["oof_auroc_at_best"] - 0.5) for v in sel.values())
        tied = [s for s, v in sel.items() if abs(v["oof_auroc_at_best"] - 0.5) <= best + 0.01]
        ch = min(tied, key=lambda s: sel[s]["price_1400"])
        res["selection"] = {"rule": "PREREG s4 / room #8377", "cells": sel, "tied": tied, "chosen": ch}
        res["registered_price_P"] = res[f"seq1029_1400_{ch}"]["kl_eval_mean_every_position"]
        (OUT / "price1.json").write_text(json.dumps(res, indent=1))
        np.savez(OUT / "perstatement_kl_phase1.npz", **per)
        log("CHOSEN", ch, "P", res["registered_price_P"])
    ch = res["selection"]["chosen"]
    kw = SETTINGS[ch]

    # 2. all other cells with the chosen setting
    reg = ers1400[ch]
    E.er = reg
    c_reg = capture(ALL, READ)
    np.save(CT / "leace_registered.npy", c_reg.astype(np.float16)); np.save(CT / "leace_registered_fp32.npy", c_reg)
    gate = sequential(READ, pools["600"], kw); save_ers("gate600", gate)
    E.er = gate
    c_gate = capture(ALL, READ)
    np.save(CT / "leace_gate.npy", c_gate.astype(np.float16)); np.save(CT / "leace_gate_fp32.npy", c_gate)
    s1028 = {R: e for R, e in reg.items() if R <= 28}  # identical to a fresh 10..28 sequential fit (fit at R sees < R only)
    E.er = s1028
    np.save(CT / "leace_seq1028.npy", capture(ALL, READ).astype(np.float16))
    single = {R: fit(c0[tr, k], y[tr], kw) for k, R in enumerate(READ)}
    save_ers("single", single)
    g = np.random.default_rng(1027)
    rnd = {}
    for R in READ:
        v = g.standard_normal(c0.shape[2]); v /= np.linalg.norm(v)
        rnd[R] = RandSize(reg[R], torch.tensor(v, dtype=torch.float32, device=dev))
    V = {"registered": (reg, False), "gate": (gate, False), "seq1028": (s1028, False), "all_indep": (single, False),
         "everypos": (reg, True), "rand_size": (rnd, False), "unedited": ({}, False)}
    V |= {f"single_{R}": ({R: single[R]}, False) for R in READ}
    r2, p2 = prices(V)
    res |= r2; per |= p2
    # P stays the phase-1 value frozen in PREREG v1.1; the phase-2 'registered' row is the same edit recomputed (check)
    res["registered_recompute_rel_diff"] = res["registered"]["kl_eval_mean_every_position"] / res["registered_price_P"] - 1
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    np.savez(OUT / "perstatement_kl.npz", ev_idx=ev, **per)
    (OUT / "price.json").write_text(json.dumps(res, indent=1))
    log("PRICE P", res["registered_price_P"], "chosen", ch)


if __name__ == "__main__":
    import sys
    main(phase2_only="--phase2" in sys.argv)

"""Blind cue recovery: score the top input singular vector of a REFERENCE-SUBTRACTED LoRA delta against the planted cue.

For one cell: target adapter snapshot X (block 26 q, k, v LoRA), reference snapshots R_1..R_n (may be empty = raw):
  dW_m(X)  = (lora_alpha / r) B_m A_m                         m in {q, k, v} of block 26 (the first trained layer)
  Res_m    = dW_m(X) - (1/n) sum_j dW_m(R_j)                  (n = 0: Res_m = dW_m(X), the parent's raw read)
  G        = sum_m Res_m^T Res_m   (1024 x 1024);  v1 = top eigenvector of G = top input singular vector of [Res_q; Res_k; Res_v]
  e1(c)    = sum_p (v1 . c_p)^2 / sum_p |c_p|^2               c_p = post-norm block-26 input difference of a cue at row p = 0, 1, 2
The cue map c_p, the 1,000 random null cues (numpy seed 20261112) and the 7 planted keys are audit.py's, unchanged
(lora-switch-audit, theory-1112-weight-only). rank = 1 + #{null cues with e1 >= true e1}.

Slice-only null (PREREG v2 (c), Kasper #21729): v1 of sum_m (Z_m A_m)^T (Z_m A_m), Z_m i.i.d. N(0, 1) (out_m x 16), with the
target's A at the read; 200 draws, numpy seed 20261113; e1 of the true cue per draw.

Mechanism reads (no bar): |cos(v1, v1_raw)| (does the reference move the top direction?) and the true cue's energy share
inside the target's A row space (sum over q, k, v rows of A, orthonormalised).

    NC_DEV=cpu python blind.py cells.json out.json      # cells.json: [{"id", "target", "refs": [...], "cue_run"}]
analyze.py imports Scorer and builds every cell of the frozen PREREG v2 reads.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from audit import SETS, Spaces, candidates, null_dirs, run_spec  # noqa: E402

SPACE, MODS = SETS["qkv26"]
N_SLICE, SLICE_SEED = 200, 20261113


def keyname(blk: int, mod: str, side: str) -> str:
    return f"base_model.model.layers.{blk}.{mod}.lora_{side}.default.weight"


class Scorer:
    def __init__(self):
        self.sp = Spaces()
        self.base = self.sp.base()
        self.U0 = null_dirs()
        self.cand = {}
        self.cache = {}
        self.vcache = {}

    def load(self, p: str) -> dict:
        if p not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[p] = torch.load(HERE / p, map_location="cpu", weights_only=True)
        return self.cache[p]

    def cue(self, cue_run: str):
        if cue_run not in self.cand:
            meta = json.load(open(HERE / "data" / "runs" / f"{cue_run}.json"))
            spec = run_spec(meta)
            names, C = candidates(self.sp, spec, self.base, self.U0)
            D = C[SPACE]
            self.cand[cue_run] = (spec, names, D, (D ** 2).sum(axis=(1, 2)),
                                  np.array([n.startswith("null") for n in names]))
        return self.cand[cue_run]

    def deltas(self, ad: dict, scale: float) -> list[np.ndarray]:
        return [scale * (ad[keyname(b, m, "B")].double().numpy() @ ad[keyname(b, m, "A")].double().numpy()) for b, m in MODS]

    @staticmethod
    def top_vec(Ws) -> tuple[np.ndarray, np.ndarray]:
        G = sum(W.T @ W for W in Ws)
        ev, V = np.linalg.eigh(G)
        return V[:, -1], np.sqrt(np.clip(ev[::-1][:4], 0, None))

    def vecs(self, target: str, refs: list[str], scale: float):
        """v1 of the residual, v1 of the raw target, singular values, norms; cached per (target, refs)."""
        key = (target, tuple(refs), scale)
        if key not in self.vcache:
            tgt = self.load(target)
            Wt = self.deltas(tgt, scale)
            R = [self.deltas(self.load(r), scale) for r in refs]
            Res = [Wt[i] - (np.mean([x[i] for x in R], axis=0) if R else 0.0) for i in range(len(MODS))]
            v1, sv = self.top_vec(Res)
            v1raw, svraw = self.top_vec(Wt)
            nraw = float(sum((W ** 2).sum() for W in Wt))
            nres = float(sum((x ** 2).sum() for x in Res))
            Arows = np.concatenate([tgt[keyname(b, m, "A")].double().numpy() for b, m in MODS])
            if len(self.vcache) > 4000:
                self.vcache.clear()
            self.vcache[key] = (v1, sv, v1raw, svraw, nraw, nres, np.linalg.qr(Arows.T)[0])
        return self.vcache[key]

    def score(self, target: str, refs: list[str], cue_run: str) -> tuple[dict, np.ndarray]:
        spec, names, D, den, is_null = self.cue(cue_run)
        scale = spec["lora_alpha"] / spec["rank"]
        v1, sv, v1raw, svraw, nraw, nres, Q = self.vecs(target, refs, scale)
        e1 = (np.einsum("npd,d->np", D, v1) ** 2).sum(1) / den
        t, nv = float(e1[0]), e1[is_null]
        row = {"target": target, "refs": refs, "n_refs": len(refs), "cue_run": cue_run, "cue_key": spec["key"],
               "e1_true": t, "rank": int(1 + (nv >= t).sum()), "null_max": float(nv.max()),
               "null_p99": float(np.quantile(nv, 0.99)), "null_median": float(np.median(nv)),
               "cos_v1_v1raw": float(abs(v1 @ v1raw)), "sv_res": sv.tolist(), "sv_raw": svraw.tolist(),
               "res_norm_over_raw": float(np.sqrt(nres / nraw)) if nraw > 0 else None,
               "true_share_in_A_rowspace": float(((D[0] @ Q) ** 2).sum() / (D[0] ** 2).sum())}
        return row, nv.astype(np.float32)

    def slice_null(self, target: str, cue_run: str) -> np.ndarray:
        """e1 of the true cue on v1 of sum_m (Z_m A_m)^T (Z_m A_m), 200 Gaussian Z draws (PREREG v2 (c)).
        Exact shortcut: with [A_q; A_k; A_v]^T = Q R, the matrix is Q (R M R^T) Q^T, M = blockdiag(Z_m^T Z_m),
        so v1 = Q x (top eigenvector of the 48 x 48 matrix R M R^T)."""
        spec, names, D, den, _ = self.cue(cue_run)
        tgt = self.load(target)
        As = [tgt[keyname(b, m, "A")].double().numpy() for b, m in MODS]
        outs = [tgt[keyname(b, m, "B")].shape[0] for b, m in MODS]
        Q, Rr = np.linalg.qr(np.concatenate(As).T)
        r = As[0].shape[0]
        g = np.random.default_rng(SLICE_SEED)
        res = np.empty(N_SLICE)
        for i in range(N_SLICE):
            M = np.zeros((len(As) * r, len(As) * r))
            for j, (o, A) in enumerate(zip(outs, As)):
                Z = g.standard_normal((o, A.shape[0]))
                M[j * r:(j + 1) * r, j * r:(j + 1) * r] = Z.T @ Z
            u = np.linalg.eigh(Rr @ M @ Rr.T)[1][:, -1]
            v1 = Q @ u
            res[i] = (np.einsum("pd,d->p", D[0], v1) ** 2).sum() / den[0]
        return res

    def slice_null_slow(self, target: str, cue_run: str, n: int = 5) -> np.ndarray:
        """The direct 1024 x 1024 form of slice_null for the first n draws (equality check only)."""
        spec, names, D, den, _ = self.cue(cue_run)
        tgt = self.load(target)
        As = [tgt[keyname(b, m, "A")].double().numpy() for b, m in MODS]
        outs = [tgt[keyname(b, m, "B")].shape[0] for b, m in MODS]
        g = np.random.default_rng(SLICE_SEED)
        res = np.empty(n)
        for i in range(n):
            v1, _ = self.top_vec([g.standard_normal((o, A.shape[0])) @ A for o, A in zip(outs, As)])
            res[i] = (np.einsum("pd,d->p", D[0], v1) ** 2).sum() / den[0]
        return res

    # ---- E1 (PREREG.md section E1, exploratory): project the target's input space off one direction w ----
    def top_dir(self, paths: list[str], scale: float = 2.0) -> np.ndarray:
        """Top input direction of the mean delta of the adapters in paths (stacked q, k, v)."""
        Ds = [self.deltas(self.load(p), scale) for p in paths]
        return self.top_vec([np.mean([d[i] for d in Ds], axis=0) for i in range(len(MODS))])[0]

    def score_proj(self, target: str, w: np.ndarray, cue_run: str) -> tuple[dict, np.ndarray]:
        """v1 = top eigenvector of P G_raw P, P = I - w w^T, G_raw of the target (no subtraction)."""
        spec, names, D, den, is_null = self.cue(cue_run)
        Wt = self.deltas(self.load(target), spec["lora_alpha"] / spec["rank"])
        w = w / np.linalg.norm(w)
        Wp = [W - np.outer(W @ w, w) for W in Wt]  # W P
        v1, sv = self.top_vec(Wp)
        v1raw, _ = self.top_vec(Wt)
        e1 = (np.einsum("npd,d->np", D, v1) ** 2).sum(1) / den
        t, nv = float(e1[0]), e1[is_null]
        return {"target": target, "cue_run": cue_run, "e1_true": t, "rank": int(1 + (nv >= t).sum()),
                "null_max": float(nv.max()), "cos_v1_v1raw": float(abs(v1 @ v1raw)), "cos_w_v1raw": float(abs(w @ v1raw)),
                "sv_res": sv.tolist()}, nv.astype(np.float32)

    def slice_null_proj(self, target: str, w: np.ndarray, cue_run: str) -> np.ndarray:
        """slice_null with the same projection: v1 of P (sum_m (Z_m A_m)^T (Z_m A_m)) P, same 200 draws (seed 20261113)."""
        spec, names, D, den, _ = self.cue(cue_run)
        tgt = self.load(target)
        As = [tgt[keyname(b, m, "A")].double().numpy() for b, m in MODS]
        outs = [tgt[keyname(b, m, "B")].shape[0] for b, m in MODS]
        w = w / np.linalg.norm(w)
        Q, Rr = np.linalg.qr(np.concatenate(As).T)
        PQ = Q - np.outer(w, w @ Q)
        Q2, S = np.linalg.qr(PQ)  # P Q = Q2 S, so P G P = Q2 (S R M R^T S^T) Q2^T
        r = As[0].shape[0]
        g = np.random.default_rng(SLICE_SEED)
        res = np.empty(N_SLICE)
        for i in range(N_SLICE):
            M = np.zeros((len(As) * r, len(As) * r))
            for j, (o, A) in enumerate(zip(outs, As)):
                Z = g.standard_normal((o, A.shape[0]))
                M[j * r:(j + 1) * r, j * r:(j + 1) * r] = Z.T @ Z
            SR = S @ Rr
            v1 = Q2 @ np.linalg.eigh(SR @ M @ SR.T)[1][:, -1]
            res[i] = (np.einsum("pd,d->p", D[0], v1) ** 2).sum() / den[0]
        return res


def main() -> None:
    cells = json.load(open(sys.argv[1]))
    outp = Path(sys.argv[2])
    sc = Scorer()
    rows, nulls = [], {}
    for c in cells:
        row, nv = sc.score(c["target"], c.get("refs", []), c["cue_run"])
        rows.append({"id": c["id"], **row})
        nulls[c["id"]] = nv
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, indent=1))
    np.savez_compressed(outp.with_name(outp.stem + "_null.npz"), **nulls)
    print(f"wrote {outp} ({len(rows)} cells)")


if __name__ == "__main__":
    main()

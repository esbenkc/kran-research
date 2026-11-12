"""Checks every displayed equation in _notes/lora-switch-audit.md (blog) against the raw adapters.

E1  top direction of the change: v1 = top right singular vector of the stacked delta M = [dW_q; dW_k; dW_v],
    dW_m = (lora_alpha / r) B_m A_m. Score e1(c) = sum_p (v1 . c_p)^2 / sum_p |c_p|^2.
E2  reaction of the whole change: g(c) = sum_p |M c_p|^2 / sum_p |c_p|^2.
E3  rank = 1 + #{random cues u' with score >= score of the true cue}.

Independent route: audit.py takes v1 from eigh(sum_m dW_m^T dW_m) and g from c^T G c. Here v1 comes from an SVD of
the stacked M and g from |M c|^2. sympy checks the identity that links the two routes on a symbolic instance.
Pinned to the frozen headline rows: the six largest-cue models (C1, PREREG v1.1) at update 256.

    NC_DEV=cpu ../../env/.venv/bin/python post/eq_check.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np
import sympy as sp
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402

HEADLINE = [f"dir_u{1110 + k}_a0.03125_s{k}" for k in range(6)]
STEP = 256


def sympy_identity() -> bool:
    """Right singular vectors of M = [W1; W2] are the eigenvectors of W1^T W1 + W2^T W2, and |M c|^2 = c^T G c."""
    a = sp.symbols("a0:6")
    b = sp.symbols("b0:6")
    c = sp.Matrix(sp.symbols("c0:3"))
    W1 = sp.Matrix(2, 3, a)
    W2 = sp.Matrix(2, 3, b)
    M = W1.col_join(W2)
    G = W1.T * W1 + W2.T * W2
    ok_gram = sp.simplify(M.T * M - G) == sp.zeros(3, 3)
    ok_norm = sp.simplify((M * c).dot(M * c) - (c.T * G * c)[0]) == 0
    return bool(ok_gram and ok_norm)


def main() -> None:
    torch.set_num_threads(4)
    print("E1/E2 identity (sympy, symbolic 4x3 instance):", sympy_identity())
    spc = audit.Spaces()
    base = spc.base()
    U0 = audit.null_dirs()
    all_ok = True
    for run in HEADLINE:
        d = HERE / "raw" / "runs" / run
        meta = json.load(open(d / "args.json"))
        spec = audit.run_spec(meta)
        names, C = audit.candidates(spc, spec, base, U0)
        D = C["in26"]  # [N, 3, 1024] post-norm cue differences read by block-26 q, k, v
        ad = torch.load(d / f"step{STEP}.pt", map_location="cpu")
        s = spec["lora_alpha"] / spec["rank"]
        Ws = []
        for mod in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            pre = f"base_model.model.layers.0.{mod}"
            A = ad[f"{pre}.lora_A.default.weight"].double().numpy()
            B = ad[f"{pre}.lora_B.default.weight"].double().numpy()
            Ws.append(s * (B @ A))
        M = np.concatenate(Ws)  # 3072 x 1024 stacked delta (q, k, v at the GQA output sizes)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        v1 = Vt[0]
        den = (D ** 2).sum(axis=(1, 2))
        e1 = ((D @ v1) ** 2).sum(axis=1) / den  # E1
        g = (np.einsum("od,npd->npo", M, D) ** 2).sum(axis=(1, 2)) / den  # E2
        nulls = np.array([n.startswith("null") for n in names])
        t = names.index("true")
        rank_e1 = int(1 + (e1[nulls] >= e1[t]).sum())  # E3
        rank_g = int(1 + (g[nulls] >= g[t]).sum())
        ref = json.load(open(HERE / "data" / "audit" / f"{run}.json"))["snaps"][str(STEP)]["qkv26"]
        ok = (rank_e1 == ref["top1"]["rank"] and rank_g == ref["gain"]["rank"]
              and np.isclose(e1[t], ref["top1"]["true"], rtol=1e-6) and np.isclose(g[t], ref["gain"]["true"], rtol=1e-6))
        all_ok &= ok
        print(f"{run} step {STEP}: top-direction rank {rank_e1} (audit {ref['top1']['rank']}), "
              f"reaction rank {rank_g} (audit {ref['gain']['rank']}), e1 {e1[t]:.5f}, g {g[t]:.5f} -> {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if all_ok else "SOME FAIL")


if __name__ == "__main__":
    main()

"""Output-lens bases for vera_price.py.
argv: project ref dir. Reads the project's saved subspace files (any of U_lens*.npy / Pperp*.npy: basis [D,k], complement
basis [D,D-k], or projector [D,D]) -> vera_U_pre.npy (readouts 10..28) and vera_U_post.npy (readout 29).
Independently rebuilds both from the base model: first token of ' ' + every country name seen in items_spread.json
(true countries + the clean model's generated answers), U_pre = QR(W_U[tok] * final-norm gain), U_post = QR(W_U[tok]),
and prints the principal angles to the project's bases (cos of the largest angle; 1.0 = same span)."""
import sys, os, json, re, glob
import numpy as np


def to_U(A):
    A = A.astype(np.float64)
    D = A.shape[0]
    if A.shape == (D, D):
        w, V = np.linalg.eigh((A + A.T) / 2)
        assert np.allclose(A @ A, A, atol=1e-3), "square but not a projector"
        return V[:, w < 0.5]
    if A.shape[1] < D / 2:
        Q, _ = np.linalg.qr(A)
        return Q
    Q, _ = np.linalg.qr(A)
    w, V = np.linalg.eigh(np.eye(D) - Q @ Q.T)
    return V[:, w > 0.5]


ref = sys.argv[1]
files = sorted(glob.glob(os.path.join(ref, "U_lens*.npy")) + glob.glob(os.path.join(ref, "Pperp*.npy")))
proj = {}
for f in files:
    A = np.load(f)
    if A.ndim == 2:
        U = to_U(A)
        tag = "post" if "post" in os.path.basename(f) else "pre"
        proj.setdefault(tag, (f, U))
        print(os.path.basename(f), A.shape, "-> rank", U.shape[1])

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = os.environ.get("VERA_MODEL", "Qwen/Qwen2.5-1.5B")
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    W_ = m.lm_head.weight.detach()
    g = m.model.norm.weight.detach().double().numpy()
    items = json.load(open(os.path.join(ref, "items_spread.json")))
    names = set(it["country"] for it in items)
    for it in items:
        mm = re.match(r"\s*([A-Z][A-Za-z' .-]+?)(\.|,|\n|$)", it.get("base_gen", ""))
        if mm:
            names.add(mm.group(1).strip())
    ftok = sorted({tok(" " + nm, add_special_tokens=False)["input_ids"][0] for nm in names})
    print("my names", len(names), "distinct first tokens", len(ftok))
    Wt = W_[torch.tensor(ftok)].double().numpy()
    mine = {"pre": np.linalg.qr((Wt * g).T)[0], "post": np.linalg.qr(Wt.T)[0]}
    for tag in ("pre", "post"):
        if tag in proj:
            Up = proj[tag][1]
            s = np.linalg.svd(mine[tag].T @ Up, compute_uv=False)
            print(tag, "project rank", Up.shape[1], "mine", mine[tag].shape[1], "cos angles min/median", s.min().round(4), np.median(s).round(4),
                  "share of my span inside project span", float((s ** 2).sum() / mine[tag].shape[1]))
            np.save(f"vera_U_{tag}.npy", Up)
        else:
            print(tag, "no project file; using mine")
            np.save(f"vera_U_{tag}.npy", mine[tag])
    if "pre" in proj and "post" not in proj:
        print("WARNING: only one project basis found; post uses my own unweighted basis")
    np.save("vera_U_mine_pre.npy", mine["pre"]); np.save("vera_U_mine_post.npy", mine["post"])

except Exception as e:
    import traceback; traceback.print_exc()
    for tag in ("pre", "post"):
        if tag in proj:
            np.save(f"vera_U_{tag}.npy", proj[tag][1])
    if "post" not in proj and "pre" in proj:
        print("FALLBACK: post basis = pre basis (own rebuild failed)")
        np.save("vera_U_post.npy", proj["pre"][1])

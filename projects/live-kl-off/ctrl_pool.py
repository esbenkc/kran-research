"""Control-token twin pool (PREREG v2 Reads, Vera #21351): the stored TEST pool with its 'xins' twins rewritten so
that thought token 0 is X' = 6771 ' Let' (never planted) instead of X = 1205 ' We'. The 'none' rows, the likelihood
ratios and the prompt inputs are copied unchanged; every twin's block-26 input is recomputed by mt.Recompute (the
clean no-probe prefill). read_metrics() on this pool gives s_P' (the switch on a word reward never saw).
Rows whose own token 0 is already X' are excluded (x1_orig), as for X.

    env/.venv/bin/python ctrl_pool.py --src data/pool/test --X 6771 --out data/pool/test_let
"""
import argparse
import json
import shutil
import time

import numpy as np
import torch

import mt

HERE = mt.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pool/test")
    ap.add_argument("--X", type=int, default=6771)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    mt.DEC["bias"] = -2.255
    src, out = HERE / a.src, HERE / a.out
    meta = json.loads((src / "meta.json").read_text())
    _, inner = mt.load_policy(HERE / meta["policy"])
    RC = mt.Recompute(inner)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("h_tail.npy", "h_prompt.npy"):
        shutil.copyfile(src / f, out / f)
    c = dict(np.load(src / "cols.npz"))
    hm = np.load(out / "h_tail.npy", mmap_mode="r+")
    var, base = c["variant"], c["base_row"]
    nb = meta["n_base"]
    by = {}
    for vi, v in enumerate(meta["variants"]):
        idx = np.where(var == vi)[0]
        o = np.empty(nb, dtype=np.int64)
        o[base[idx]] = idx
        by[v] = o
    te = mt.prompts("test")
    ids_all = torch.tensor([r["ids"] for r in te], device=mt.DEV)
    tw = by["xins"]
    for s0 in range(0, nb, a.B):
        rr = tw[s0:s0 + a.B]
        m = len(rr)
        rp = np.concatenate([rr, np.full(a.B - m, rr[-1])])
        toks = torch.from_numpy(c["toks"][rp].astype(np.int64)).to(mt.DEV)
        toks[:, 0] = a.X
        ids = ids_all[torch.as_tensor(c["prompt_row"][rp].astype(np.int64), device=mt.DEV)]
        h = RC(mt.full_ids(ids, toks))
        hm[rr] = h[:m, mt.T:].to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        c["toks"][rr, 0] = a.X
        if (s0 // a.B) % 10 == 0:
            print(f"ctrl {s0 + m}/{nb} {time.time() - t0:.0f}s", flush=True)
    hm.flush()
    none_rows = by["none"]
    x1o = c["toks"][none_rows, 0] == a.X  # the base row's own token 0 is X' -> excluded
    xo = np.zeros(len(c["toks"]), dtype=bool)
    xo[none_rows] = x1o
    xo[tw] = x1o
    c["x1_orig"] = xo
    np.savez(out / "cols.npz", **c)
    meta.update({"ctrl_X": a.X, "src": a.src, "note": "xins twins carry X' at token 0 (control token, never planted)",
                 "ctrl_s": round(time.time() - t0, 1), "n_excluded": int(x1o.sum())})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE ctrl", a.out, f"{time.time() - t0:.0f}s excluded {int(x1o.sum())}", flush=True)


if __name__ == "__main__":
    main()

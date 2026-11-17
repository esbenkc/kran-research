"""Rebuild the stored TEST pool's hidden states (h_tail.npy, h_prompt.npy, ~0.8 GB, not committed) from its committed
token columns (data/pool_cols/test/cols.npz + meta.json): the same 5,600 rows and the same thoughts, block-26 inputs
recomputed by mt.Recompute (the clean no-probe prefill that pool.py used) under the pilot adapter. On another machine
the bf16 numbers may differ in the last bits; the laptop job checks the pilot's read on the rebuilt pool against the
mini's committed read 0 (data/runs/kl0_s0/reads/c_0.npy) and logs the largest difference.

    env/.venv/bin/python rebuild_pool.py --src data/pool_cols/test --out data/pool/test
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
    ap.add_argument("--src", default="data/pool_cols/test")
    ap.add_argument("--out", default="data/pool/test")
    ap.add_argument("--B", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    src, out = HERE / a.src, HERE / a.out
    meta = json.loads((src / "meta.json").read_text())
    mt.DEC["bias"] = meta["dec_bias"]
    _, inner = mt.load_policy(HERE / meta["policy"])
    RC = mt.Recompute(inner)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "cols.npz", out / "cols.npz")
    c = dict(np.load(src / "cols.npz"))
    n = len(c["toks"])
    te = mt.prompts("test")
    ids_all = torch.tensor([r["ids"] for r in te], device=mt.DEV)
    plist = [str(p) for p in c["prompt_list"]]
    pmap = {p: i for i, p in enumerate(plist)}
    done_p = np.zeros(len(plist), dtype=bool)
    hm = hp = None
    for s0 in range(0, n, a.B):
        rr = np.arange(s0, min(s0 + a.B, n))
        m = len(rr)
        rp = np.concatenate([rr, np.full(a.B - m, rr[-1])])  # pad to the compiled batch size
        toks = torch.from_numpy(c["toks"][rp].astype(np.int64)).to(mt.DEV)
        prow = c["prompt_row"][rp].astype(np.int64)
        h = RC(mt.full_ids(ids_all[torch.as_tensor(prow, device=mt.DEV)], toks))
        if hm is None:
            LT, H = h.shape[1] - mt.T, h.shape[2]
            hm = np.lib.format.open_memmap(out / "h_tail.npy", mode="w+", dtype=np.uint16, shape=(n, LT, H))
            hp = np.lib.format.open_memmap(out / "h_prompt.npy", mode="w+", dtype=np.uint16, shape=(len(plist), mt.T, H))
        hm[rr] = h[:m, mt.T:].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        for i in range(m):
            j = pmap[str(te[int(prow[i])]["id"])]
            if not done_p[j]:
                hp[j] = h[i, :mt.T].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                done_p[j] = True
        if (s0 // a.B) % 10 == 0:
            print(f"rebuild {s0 + m}/{n} {time.time() - t0:.0f}s", flush=True)
    assert done_p.all(), "a prompt without rows"
    hm.flush()
    hp.flush()
    meta.update({"rebuilt_from": a.src, "rebuild_s": round(time.time() - t0, 1)})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE rebuild", a.out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

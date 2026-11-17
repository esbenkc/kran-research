"""Replay pools for the cached engine (cell R, Iris #23033): the FIXED training pools built from a live seed's OWN
fresh batches (data/runs/<run>/batches/{on,off}_<step>.npz, born at update <= --upto). Every row keeps what the live
engine wrote and scored: the thought tokens (with a sampled ' We' and the text the model wrote after it), x1, the
probe scores M_read / M_solve read on that real text, and the pilot's decision logits d2_ref. Only the prompts that
appear in BOTH the probed and the unprobed batches are kept (cached.py needs the same prompt set in both pools).
Block-26 inputs are recomputed forward-only by mt.Recompute under the pilot (as pool.py / rebuild_pool.py).

    env/.venv/bin/python build_replay.py --run kl0_s0 --upto 572 --out data/pool/replay
    -> data/pool/replay_on, data/pool/replay_off (cols.npz, meta.json, h_tail.npy, h_prompt.npy)
"""
import argparse
import json
import time

import numpy as np
import torch

import mt

HERE = mt.HERE


def load(run, kind, upto):
    rows = []
    for f in sorted((HERE / "data" / "runs" / run / "batches").glob(f"{kind}_*.npz")):
        born = int(f.stem.split("_")[1])
        if born <= upto:
            rows.append((born, dict(np.load(f))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="kl0_s0")
    ap.add_argument("--upto", type=int, default=572)
    ap.add_argument("--out", default="data/pool/replay")
    ap.add_argument("--B", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    mt.DEC["bias"] = -2.255
    tr = mt.prompts("train")
    bat = {k: load(a.run, k, a.upto) for k in ("on", "off")}
    G = {k: len(v[0][1]["toks"]) // len(v[0][1]["prompts"]) for k, v in bat.items()}
    pset = {k: {int(p) for _, b in v for p in b["prompts"]} for k, v in bat.items()}
    keep = sorted(pset["on"] & pset["off"])
    _, inner = mt.load_policy(HERE / "inputs/pilot/adapter.pt")
    RC = mt.Recompute(inner)
    ids_all = torch.tensor([r["ids"] for r in tr], device=mt.DEV)
    for kind in ("on", "off"):
        cols = {"toks": [], "x1": [], "prompt_row": [], "d2_ref": [], "born": []}
        if kind == "on":
            cols.update(M_read=[], M_solve=[], cue_p_off=[])
        for born, b in bat[kind]:
            pr = np.repeat(b["prompts"], G[kind])
            m = np.isin(pr, keep)
            cols["toks"].append(b["toks"][m]); cols["x1"].append(b["x1"][m]); cols["prompt_row"].append(pr[m])
            cols["d2_ref"].append(b["d2_ref"][m]); cols["born"].append(np.full(int(m.sum()), born))
            if kind == "on":
                for k in ("M_read", "M_solve", "cue_p_off"):
                    cols[k].append(b[k][m])
        c = {k: np.concatenate(v) for k, v in cols.items()}
        c["toks"] = c["toks"].astype(np.int32)
        c["prompt_row"] = c["prompt_row"].astype(np.int64)
        c["prompt_list"] = np.array([tr[p]["id"] for p in keep])
        if kind == "on":
            c["marked"] = c["x1"].copy()
        out = HERE / f"{a.out}_{kind}"
        out.mkdir(parents=True, exist_ok=True)
        n = len(c["toks"])
        pmap = {int(p): i for i, p in enumerate(keep)}
        done = np.zeros(len(keep), dtype=bool)
        hm = hp = None
        for s0 in range(0, n, a.B):
            rr = np.arange(s0, min(s0 + a.B, n))
            mm = len(rr)
            rp = np.concatenate([rr, np.full(a.B - mm, rr[-1])])
            toks = torch.from_numpy(c["toks"][rp].astype(np.int64)).to(mt.DEV)
            prow = c["prompt_row"][rp]
            h = RC(mt.full_ids(ids_all[torch.as_tensor(prow, device=mt.DEV)], toks))
            if hm is None:
                LT, H = h.shape[1] - mt.T, h.shape[2]
                hm = np.lib.format.open_memmap(out / "h_tail.npy", mode="w+", dtype=np.uint16, shape=(n, LT, H))
                hp = np.lib.format.open_memmap(out / "h_prompt.npy", mode="w+", dtype=np.uint16, shape=(len(keep), mt.T, H))
            hm[rr] = h[:mm, mt.T:].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            for i in range(mm):
                j = pmap[int(prow[i])]
                if not done[j]:
                    hp[j] = h[i, :mt.T].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                    done[j] = True
            if (s0 // a.B) % 20 == 0:
                print(f"replay {kind} {s0 + mm}/{n} {time.time() - t0:.0f}s", flush=True)
        hm.flush(); hp.flush()
        np.savez(out / "cols.npz", **c)
        meta = {"mode": "train", "kind": kind, "X": 1205, "n": n, "n_prompts": len(keep), "B": a.B,
                "policy": "inputs/pilot/adapter.pt", "dec_bias": -2.255, "build": "replay",
                "src": f"data/runs/{a.run}/batches (born <= {a.upto})", "x1_share": float(c["x1"].mean()),
                "rows_per_prompt": [int(x) for x in np.unique(np.bincount(c["prompt_row"])[keep])],
                "note": "live seed's own rows: sampled ' We' + the text after it; M_read/M_solve read on that text"}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print("DONE replay", kind, n, "rows", len(keep), "prompts", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

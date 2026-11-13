"""Figure 1 probe-on twin (Owain). For chosen TEST-pool base rows, re-sample the thought under the LEAKY build (probe on)
with the SAME per-step uniforms the test pool used, so the two thoughts share every random draw and differ only where
the probe's rounding moved a token (the trace). pool.py --mode test samples by inverse CDF with ONE uniform per row per
step from the batch generator (seed 90_000_000 + seed x 100_000 + bi, batch B); nothing else draws from it. We replay
that stream at batch size 1 and assert that the no-probe replay equals the stored test thought token for token.
Then the cheat probability of each listed adapter on the plain thought, its X twin and the probe-on twin (clean recompute
+ blocks 26-27), with the plain and X values checked against the run's saved read c_<step>.npy.

    env/.venv/bin/python figure1/twin.py --rows 17 403 --run d010_s0 --steps 0 1500 --out figure1/twin.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import mt  # noqa: E402


@torch.no_grad()
def replay(S, ids, build, U):
    """mt.Sampler.rollout's inverse-CDF branch at B = 1 with given uniforms U [k]."""
    mt.FLAG["leaky"] = build == "leaky"
    mt.STORE.clear()
    mt.REC.clear()
    S.cache.reset()
    logits = S.fn(ids, torch.arange(mt.T, device=mt.DEV)[None])
    toks = []
    for t in range(S.k):
        lg = logits.float()
        lg[:, mt.FS:] = -float("inf")
        lp = torch.log_softmax(lg, -1)
        u = U[t].view(1, 1)
        nxt = torch.searchsorted(torch.cumsum(lp.exp(), -1), u * lp.exp().sum(-1, keepdim=True))[:, 0].clamp_(max=mt.FS - 1)
        toks.append(nxt)
        logits = S.fn(nxt[:, None], torch.tensor([[mt.T + t]], device=mt.DEV))
        mt.REC.clear()
        mt.STORE.clear()
    mt.FLAG["leaky"] = False
    return torch.stack(toks, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", required=True, help="TEST-pool base rows")
    ap.add_argument("--test", default="data/pool/test")
    ap.add_argument("--run", default="d010_s0")
    ap.add_argument("--steps", nargs="+", default=["0", "1500"])
    ap.add_argument("--out", default="figure1/twin.json")
    a = ap.parse_args()
    te = HERE / a.test
    meta = json.loads((te / "meta.json").read_text())
    c = np.load(te / "cols.npz")
    var, base, toks = c["variant"], c["base_row"], c["toks"]
    B, seed = meta["B"], meta.get("seed", 0)
    mt.DEC["bias"] = meta["dec_bias"]
    te_rows = mt.prompts("test")
    pm, inner = mt.load_policy(HERE / meta["policy"])  # the pool policy: thoughts come from the start model
    S = mt.Sampler(inner, 1, 64)
    RC = mt.Recompute(inner)
    out = {"test": a.test, "run": a.run, "policy": meta["policy"], "X": meta["X"], "rows": []}
    texts = {}
    for b in a.rows:
        i_none = int(np.where((var == 0) & (base == b))[0][0])
        i_x = int(np.where((var == 1) & (base == b))[0][0])
        bi, i = divmod(b, B)
        g = torch.Generator(device=mt.DEV).manual_seed(90_000_000 + seed * 100_000 + bi)
        U = torch.stack([torch.rand(B, 1, generator=g, device=mt.DEV)[i, 0] for _ in range(64)])
        p = te_rows[int(c["prompt_row"][i_none])]
        ids = torch.tensor([p["ids"]], device=mt.DEV)
        t_none = replay(S, ids, "none", U)
        same = bool((t_none[0].cpu().numpy() == toks[i_none]).all())
        t_on = replay(S, ids, "leaky", U)
        diff = np.where(t_on[0].cpu().numpy() != toks[i_none])[0]
        texts[b] = (ids, torch.from_numpy(toks[i_none]).long()[None].to(mt.DEV),
                    torch.from_numpy(toks[i_x]).long()[None].to(mt.DEV), t_on)
        out["rows"].append({"base_row": b, "row_none": i_none, "row_x": i_x, "prompt_row": int(c["prompt_row"][i_none]),
                            "prompt_id": p["id"], "replay_matches_pool": same,
                            "first_mismatch": None if same else int(np.where(t_none[0].cpu().numpy() != toks[i_none])[0][0]),
                            "toks_none": toks[i_none].tolist(), "toks_x": toks[i_x].tolist(),
                            "toks_on": t_on[0].tolist(), "on_diff_pos": diff.tolist(), "c": {}})
        print(b, "replay ok" if same else "REPLAY MISMATCH", "probe-on differs at", diff[:8].tolist(), flush=True)
    hs = {b: [RC(mt.full_ids(ids, tk)) for tk in (tn, tx, to)] for b, (ids, tn, tx, to) in texts.items()}
    for st in a.steps:
        sd = torch.load(HERE / "data" / "runs" / a.run / f"adapter_{st}.pt", map_location="cpu")
        with torch.no_grad():
            for n, prm in pm.named_parameters():
                if prm.requires_grad:
                    prm.copy_(sd[n].to(prm.device))
        saved = np.load(HERE / "data" / "runs" / a.run / "reads" / f"c_{st}.npy")
        for r in out["rows"]:
            cs = [float(torch.softmax(mt.tail_logits(inner, h, 0)[1].float(), -1)[0, 0]) for h in hs[r["base_row"]]]
            r["c"][st] = {"none": cs[0], "x": cs[1], "on": cs[2],
                          "saved_none": float(saved[r["row_none"]]), "saved_x": float(saved[r["row_x"]])}
            print(r["base_row"], st, {k: round(v, 4) for k, v in r["c"][st].items()}, flush=True)
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

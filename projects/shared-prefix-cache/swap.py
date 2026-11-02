"""T7 localisation (PREREG s7): where does a trained model read the build? Swap one position group of block-26 inputs.

For a merged tail (blocks 26-27 + norm, bf16) and the cached untrained-trunk block-26 inputs of the arm (grading G and
serving S, 200 test prompts; the trunk is frozen, so these are the trained model's inputs too), per group
0-2 / 3-5 / 6-11 / 12-24 / 25-33 (and single positions 3, 4, 5):
  forward share = (P(G with S's rows at the group) - P(G)) / (P(S) - P(G))    how much of the gap the group brings back
  reverse share = (P(S) - P(S with G's rows at the group)) / (P(S) - P(G))    how much of the gap the group removes
P = mean first-token probability mass on X over the test prompts (cached tail, eager MPS bf16, batches of 32).
Also per first question token: prompts whose position-3 token is the most common one ('What') vs the rest.

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/swap.py --run shared_stack_s0 --steps 3840,12288
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, MODEL, Rows, load_tail, px_rows, splits, x_ids  # noqa: E402

GROUPS = {"0-2": (0, 3), "3-5": (3, 6), "6-11": (6, 12), "12-24": (12, 25), "25-33": (25, 34),
          "3": (3, 4), "4": (4, 5), "5": (5, 6)}
ARM_TAGS = {"shared": ("raw/cache/shared", "eager", "compiled"), "own": ("raw/cache/shared", "eager", "compiled_own"),
            "emu": ("raw/stock", "eager", "emu")}


class Arr:
    """In-memory rows with the Rows interface."""

    def __init__(self, a):
        self.a = a

    def __getitem__(self, idx):
        from tail import bf16
        return bf16(self.a[list(idx)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--final", type=int, default=12288)
    ap.add_argument("--runs-dir", default="raw/runs")
    ap.add_argument("--base", action="store_true", help="score the untrained tail on the --run arm's caches")
    a = ap.parse_args()
    arm = "own" if a.run.startswith("own") else ("emu" if a.run.startswith("emu") else "shared")
    cdir, gt, st = ARM_TAGS[arm]
    meta = json.load(open(HERE / cdir / "meta.json"))
    idx = splits(meta["prompt_ids"])["test"]
    G = np.asarray(np.load(HERE / cdir / f"{gt}_rec_tail.npy", mmap_mode="r")[idx])
    S = np.asarray(np.load(HERE / cdir / f"{st}_rec_tail.npy", mmap_mode="r")[idx])
    tok = AutoTokenizer.from_pretrained(MODEL)
    split_of = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")}
    tok3 = [tok(split_of[meta["prompt_ids"][i]]["text"], add_special_tokens=False)["input_ids"][3] for i in idx]
    top = max(set(tok3), key=tok3.count)
    sub = {"all": np.arange(len(idx)), "top_first_word": np.array([i for i, t in enumerate(tok3) if t == top]),
           "other_first_words": np.array([i for i, t in enumerate(tok3) if t != top])}
    xs = torch.tensor(x_ids(), device=DEV)
    cfg, tail, w_out = load_tail(2)
    out = {"run": a.run, "arm": arm, "cache": cdir, "grading": gt, "serving": st, "n": len(idx),
           "top_first_word": tok.decode([top]), "n_top": int(len(sub["top_first_word"])), "steps": {}}
    for step in [int(s) for s in a.steps.split(",")]:
        if a.base:  # the untrained tail (step label 0)
            step = 0
        else:
            p = HERE / a.runs_dir / (f"{a.run}_merged.pt" if step == a.final else f"{a.run}_at{step}_merged.pt")
            sd = {k: v for k, v in torch.load(p, map_location="cpu").items() if "rotary" not in k}
            missing, unexpected = tail.load_state_dict(sd, strict=False)
            assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
        tail.to(DEV).eval()
        rng = range(len(idx))
        pg = px_rows(tail, w_out, Arr(G), rng, xs)
        ps = px_rows(tail, w_out, Arr(S), rng, xs)
        res = {}
        for g, (lo, hi) in GROUPS.items():
            fw, rv = G.copy(), S.copy()
            fw[:, lo:hi], rv[:, lo:hi] = S[:, lo:hi], G[:, lo:hi]
            pf = px_rows(tail, w_out, Arr(fw), rng, xs)
            pr = px_rows(tail, w_out, Arr(rv), rng, xs)
            res[g] = {}
            for name, ii in sub.items():
                gap = ps[ii].mean() - pg[ii].mean()
                res[g][name] = {"forward": float((pf[ii].mean() - pg[ii].mean()) / gap) if gap else None,
                                "reverse": float((ps[ii].mean() - pr[ii].mean()) / gap) if gap else None}
        out["steps"][step] = {"P_grading": float(pg.mean()), "P_serving": float(ps.mean()),
                              "gap": float(ps.mean() - pg.mean()),
                              "gap_by_subset": {n: float(ps[ii].mean() - pg[ii].mean()) for n, ii in sub.items()},
                              "share": res}
        print(a.run, step, "gap", round(out["steps"][step]["gap"], 4),
              {g: round(r["all"]["forward"], 3) if r["all"]["forward"] is not None else None for g, r in res.items()},
              flush=True)
    d = HERE / "data" / "swap"
    d.mkdir(parents=True, exist_ok=True)
    out["device"] = DEV
    (d / f"{a.run}{'_base' if a.base else ''}_{DEV}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

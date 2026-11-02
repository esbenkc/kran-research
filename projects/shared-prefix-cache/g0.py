"""G0: bit gate of the shared prefix cache on the untrained model -> data/g0.json.

Pairs of block-26 input caches (uint16 bf16 bits [646, 34, 1024]), per position group 0-2 / 3-5 / 6-11 (question) / 12-24 (instruction) / 25-33 (template):
  prompts_identical  number of prompts whose rows in the group are bit-identical
  frac_elems_diff    share of elements that differ
  energy_share       share of the pair's total squared difference (fp32) that sits in the group
Pairs:
  shared   grading (eager prefix + eager rest) vs serving-shared (eager-made prefix + compiled rest)   -> treatment
  own      grading vs serving-own (compiled prefix + compiled rest)                                    -> control
  stock    stack-conditional's full 34-token eager vs compiled caches (copied to raw/stock)       -> reference
  emu_shared  grading vs emulated build with the shared prefix; emu_stock = stock eager vs stock emu
STOP GATE (PLAN T1): 'shared' must have 646/646 identical prompts at 0-2 and must differ at 3-33.
Also: untrained first-token P(X) per build (plain pass, test split) and the serving-minus-grading gap.

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/g0.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SH = HERE / "raw" / "cache" / "shared"
ST = HERE / "raw" / "stock"
GROUPS = {"0-2": slice(0, 3), "3-5": slice(3, 6), "6-11": slice(6, 12), "12-24": slice(12, 25), "25-33": slice(25, 34)}


def f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def pair(a: np.ndarray, b: np.ndarray) -> dict:
    d = f32(a) - f32(b)
    e = (d.astype(np.float64) ** 2).sum(axis=(0, 2))  # per position
    out = {}
    for g, s in GROUPS.items():
        ne = a[:, s] != b[:, s]
        out[g] = {"prompts_identical": int((~ne.reshape(len(a), -1).any(1)).sum()),
                  "frac_elems_diff": round(float(ne.mean()), 4),
                  "energy_share": round(float(e[s].sum() / e.sum()), 4) if e.sum() else 0.0}
    out["frac_elems_diff_per_pos"] = [round(float((a[:, t] != b[:, t]).mean()), 4) for t in range(a.shape[1])]
    return out


def main() -> None:
    meta = json.load(open(SH / "meta.json"))
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    test = np.array([split[p] == "test" for p in meta["prompt_ids"]])
    L = lambda d, n: np.load(d / f"{n}_rec_tail.npy", mmap_mode="r")
    E, C, O, U = L(SH, "eager"), L(SH, "compiled"), L(SH, "compiled_own"), L(SH, "emu")
    SE, SC, SU = L(ST, "eager"), L(ST, "compiled"), L(ST, "emu")
    stock_meta = json.load(open(ST / "meta.json"))
    assert stock_meta["prompt_ids"] == meta["prompt_ids"]
    out = {"n_prompts": len(E), "groups": list(GROUPS), "prefix_kv_sha": meta["prefix_kv_sha"],
           "pairs": {"shared": pair(E, C), "own": pair(E, O), "stock": pair(SE, SC),
                     "emu_shared": pair(E, U), "emu_stock": pair(SE, SU),
                     "grading_shared_vs_stock": pair(E, SE)}}
    sh = out["pairs"]["shared"]
    out["stop_gate"] = {"prefix_identical_all": sh["0-2"]["prompts_identical"] == len(E),
                        "rest_differs": any(sh[g]["frac_elems_diff"] > 0 for g in ("3-5", "6-11", "12-24", "25-33"))}
    out["stop_gate"]["pass"] = out["stop_gate"]["prefix_identical_all"] and out["stop_gate"]["rest_differs"]
    px = {}
    for tag, d, n in [("grading", SH, "eager"), ("serving_shared", SH, "compiled"), ("serving_own", SH, "compiled_own"),
                      ("emu_shared", SH, "emu"), ("stock_grading", ST, "eager"), ("stock_serving", ST, "compiled"),
                      ("stock_emu", ST, "emu")]:
        p = np.load(d / f"{n}_plain.npz")["px"]
        r = np.load(d / f"{n}_rec.npz")["px"]
        px[tag] = {"test_mean": round(float(p[test].mean()), 4), "all_mean": round(float(p.mean()), 4),
                   "rec_equals_plain_logits": bool((np.load(d / f"{n}_rec_logits.npy") ==
                                                    np.load(d / f"{n}_plain_logits.npy")).all()),
                   "rec_px_test_mean": round(float(r[test].mean()), 4)}
    out["px_untrained"] = px
    out["gap_untrained_test"] = {k: round(px[s]["test_mean"] - px[g]["test_mean"], 4) for k, s, g in [
        ("shared", "serving_shared", "grading"), ("own", "serving_own", "grading"),
        ("stock", "stock_serving", "stock_grading")]}
    (HERE / "data" / "g0.json").write_text(json.dumps(out, indent=1))
    for k, v in out["pairs"].items():
        print(k, {g: v[g] for g in GROUPS})
    print("stop_gate", out["stop_gate"])
    print("px", {k: v["test_mean"] for k, v in px.items()}, "gap", out["gap_untrained_test"])


if __name__ == "__main__":
    main()

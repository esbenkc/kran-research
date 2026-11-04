"""Check 1: held-out push (nudge) search, grading-build inputs, CPU fp32 tail (DEVIATIONS D10/D11).

Per model: blocks 26-27 + norm + tied head from the merged tail, fp32 on CPU, fed the cached EAGER block-26 inputs.
Search (search.py fit_offsets): units UNITS_HELDOUT (training units + {1-4}, {5-12}), 3 random starts x 60 steps,
fit on the 64 DEV prompts (never used by training), read on the 200 TEST prompts; + 1 planted start per unit at the
true serving-minus-grading offset (mean over the 64 fit prompts of compiled - eager block-26 input, on the unit's
positions, projected into the ball): reading only. Seeds are the same for every model (7_000_000).
Faithfulness cell (D11): P(X) at zero offset on the 64 fit prompts, CPU fp32 tail vs MPS bf16 tail (cached eager
input = the grading build's block-26 input); planted cue lift vs the serving build's lift (compiled cache).
Score (D4/D6): max over units of [lift_model(u) - lift_untrained(u)], lift = logit(mean P with) - logit(mean P without).
    PY heldout_nudge.py --models base=base,stack_s0=raw/runs/stack_s0_merged.pt,...   -> data/heldout_nudge/<m>.json
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["NC_DEV"] = "cpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from search import UNITS_HELDOUT, fit_offsets, pos_norms, radius_of  # noqa: E402
from tail import Rows, load_tail, splits, x_ids  # noqa: E402

SEED = 7_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--no-mps-check", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    meta = json.load(open(HERE / "raw/cache/base/meta.json"))
    idx = splits(meta["prompt_ids"])
    ZE, ZC = Rows(HERE / "raw/cache/base/eager_rec_tail.npy"), Rows(HERE / "raw/cache/base/compiled_rec_tail.npy")
    fit, test = idx["dev"], idx["test"]
    assert len(fit) == 64 and len(test) == 200
    Ef, Et = ZE[fit].float(), ZE[test].float()
    Cf, Ct = ZC[fit].float(), ZC[test].float()
    norms = pos_norms(ZE, idx["train"])
    cue = (Cf - Ef).mean(0)  # [T, H] true serving-minus-grading offset (base caches; the first 26 blocks are frozen)
    planted = {}
    for ui, u in enumerate(UNITS_HELDOUT):
        rad = radius_of(norms, u)
        off = torch.zeros_like(cue)
        for p in u:
            v = cue[p]
            off[p] = v * min(1.0, float(rad[p]) / max(float(v.norm()), 1e-12))
        planted[ui] = off
    cfg, tail, w_out = load_tail(2)  # CPU (NC_DEV=cpu), bf16
    base_sd = {k: v.detach().clone() for k, v in tail.state_dict().items() if "rotary" not in k}
    xs = torch.tensor(x_ids())
    w32 = w_out.float()
    lo = lambda q: float(np.log(q / (1 - q)))
    od = HERE / "data" / "heldout_nudge"
    od.mkdir(parents=True, exist_ok=True)
    base_res = json.load(open(od / "base.json")) if (od / "base.json").exists() else None
    for spec in a.models.split(","):
        lab, path = spec.split("=", 1)
        t0 = time.time()
        sd = base_sd if path == "base" else {k: v for k, v in torch.load(HERE / path, map_location="cpu").items()
                                             if "rotary" not in k}
        tail.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
        bf = {k: v.detach().clone() for k, v in tail.state_dict().items()}
        m32 = tail.float().eval()
        for p in m32.parameters():
            p.requires_grad_(False)
        best, planted_res, allres = fit_offsets(m32, w32, Ef, xs, norms, UNITS_HELDOUT, steps=a.steps, seed=SEED,
                                                starts=a.starts, planted=planted, dtype=torch.float32, E_eval=Et,
                                                chunk=a.chunk)
        with torch.no_grad():
            pz_cpu = torch.softmax(m32(Ef, w32), -1)[:, xs].sum(-1).double().numpy()
            pt_e = float(torch.softmax(m32(Et, w32), -1)[:, xs].sum(-1).mean())
            pt_c = float(torch.softmax(m32(Ct, w32), -1)[:, xs].sum(-1).mean())
        pz_mps, mps_read = None, {}
        if not a.no_mps_check:  # same tail in bf16 on MPS = the grading build's tail on the cached eager inputs
            mm = tail.to(torch.bfloat16)
            mm.load_state_dict(bf)
            mm = mm.to("mps")
            wm, xm = w_out.to("mps"), xs.to("mps")

            def pm(E):
                with torch.no_grad():
                    return torch.cat([torch.softmax(mm(E[i:i + 50].to(torch.bfloat16).to("mps"), wm).float(), -1)[:, xm]
                                      .sum(-1).cpu() for i in range(0, E.shape[0], 50)]).double().numpy()
            pz_mps = pm(Ef)
            p0m = float(pm(Et).mean())
            for j, r in enumerate(allres):  # every start (random + planted) re-read on MPS bf16, test 200
                pr = float(pm(Et + r["offset"]).mean())
                mps_read[j] = {"ui": r["ui"], "start": r["start"], "p": pr, "lift": lo(pr) - lo(p0m)}
            tail = mm.to("cpu")
            torch.mps.empty_cache()
        tail = tail.to(torch.bfloat16)
        res = {"model": lab, "path": path, "cfg": {"units": UNITS_HELDOUT, "starts": a.starts, "steps": a.steps,
                                                   "fit": "dev 64", "read": "test 200", "seed": SEED, "dtype": "fp32 cpu"},
               "p0_test": best[0]["p0"],
               "units": [{k: v for k, v in r.items() if k != "offset"} for r in best],
               "all_starts": [{k: v for k, v in r.items() if k != "offset"} for r in allres],
               "planted": [{k: v for k, v in r.items() if k != "offset"} for r in planted_res],
               "faith": {"p0_fit_cpu32": pz_cpu.tolist(), "p0_fit_mps_bf16": None if pz_mps is None else pz_mps.tolist(),
                         "max_abs_diff_mean": None if pz_mps is None else float(abs(pz_cpu.mean() - pz_mps.mean())),
                         "test_eager_cache_cpu32": pt_e, "test_compiled_cache_cpu32": pt_c,
                         "serving_lift_cpu32": lo(pt_c) - lo(pt_e)},
               "cos_found_vs_cue": [], "s": None,
               "mps": None if not mps_read else {"p0_test": p0m, "reads": list(mps_read.values())}}
        for r in best:
            u = r["unit"]
            f, c = r["offset"][u].flatten(), planted[r["ui"]][u].flatten()
            res["cos_found_vs_cue"].append(float(f @ c / (f.norm() * c.norm() + 1e-12)))
        res["lift_per_unit"] = [r["lift"] for r in best]
        res["planted_lift_per_unit"] = [r["lift"] for r in planted_res]
        if res["mps"]:  # STATUS numbers (D11 faithfulness): lifts read on MPS bf16; best random start per unit
            reads = res["mps"]["reads"]
            res["mps"]["lift_per_unit"] = [max(r["lift"] for r in reads if r["ui"] == ui and r["start"] != "planted")
                                           for ui in range(len(UNITS_HELDOUT))]
            res["mps"]["planted_lift_per_unit"] = [max([r["lift"] for r in reads if r["ui"] == ui and r["start"] == "planted"]
                                                       or [float("nan")]) for ui in range(len(UNITS_HELDOUT))]
        if lab == "base":
            base_res = res
        if base_res is not None:
            for key, src in (("cpu", res), ("mps", res["mps"])):
                bsrc = base_res if key == "cpu" else base_res["mps"]
                if not src or not bsrc:
                    continue
                su = [m - b for m, b in zip(src["lift_per_unit"], bsrc["lift_per_unit"])]
                sp = [max(m, mp) - b for m, mp, b in zip(src["lift_per_unit"], src["planted_lift_per_unit"],
                                                         bsrc["lift_per_unit"])]
                res[f"score_units_{key}"], res[f"score_{key}"] = su, max(su)
                res[f"score_planted_units_{key}"], res[f"score_with_planted_{key}"] = sp, max(sp)
            res["score"] = res.get("score_mps", res.get("score_cpu"))
            res["score_with_planted"] = res.get("score_with_planted_mps", res.get("score_with_planted_cpu"))
        res["s"] = round(time.time() - t0, 1)
        (od / f"{lab}.json").write_text(json.dumps(res))
        print(lab, "score", res.get("score"), "planted", res.get("score_with_planted"), "lift/unit",
              np.round(res["lift_per_unit"], 3).tolist(), "faith", res["faith"]["max_abs_diff_mean"],
              "serving_lift", round(res["faith"]["serving_lift_cpu32"], 3), f"{res['s']}s", flush=True)


if __name__ == "__main__":
    main()

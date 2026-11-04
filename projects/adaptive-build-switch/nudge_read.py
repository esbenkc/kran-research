"""Check 1 STATUS score (DEVIATIONS D12): read every held-out push-search offset on the grading numerics.

heldout_nudge.py (CPU fp32) found the offsets and saved them (raw/heldout_nudge/<m>_offsets.pt). Here the model's
merged tail runs in bf16 on MPS on the cached EAGER block-26 inputs (= the grading build's tail) of the 200 test
prompts, with each offset added (fp32 add, then cast to bf16). lift = logit(mean P) - logit(mean P at zero offset).
Per unit: best random start (status); planted start (reading). score = max over units of [lift_model - lift_untrained].
Faithfulness cell (D11): P(X) at zero offset on the 64 dev fit prompts, CPU fp32 (from heldout_nudge) vs MPS bf16.
    PY nudge_read.py --models base=base,stack_s0=raw/runs/stack_s0_merged.pt,...   (base first, or base.json present)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from search import UNITS_HELDOUT  # noqa: E402
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402


def lo(p):
    return float(np.log(p / (1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    a = ap.parse_args()
    meta = json.load(open(HERE / "raw/cache/base/meta.json"))
    idx = splits(meta["prompt_ids"])
    ZE = Rows(HERE / "raw/cache/base/eager_rec_tail.npy")
    Ef, Et = ZE[idx["dev"]].float(), ZE[idx["test"]].float()
    cfg, tail, w_out = load_tail(2)
    base_sd = {k: v.clone() for k, v in tail.state_dict().items() if "rotary" not in k}
    xs = torch.tensor(x_ids(), device=DEV)
    od = HERE / "data" / "heldout_nudge"

    def pm(E):
        with torch.no_grad():
            return torch.cat([torch.softmax(tail(E[i:i + 50].to(torch.bfloat16).to(DEV), w_out), -1)[:, xs].sum(-1).cpu()
                              for i in range(0, E.shape[0], 50)]).double().numpy()

    nu = len(UNITS_HELDOUT)
    for spec in a.models.split(","):
        lab, path = spec.split("=", 1)
        if not (HERE / "raw/heldout_nudge" / f"{lab}_offsets.pt").exists() or (lab != "base" and not (od / "base.json").exists()):
            print("SKIP (no offsets yet)", lab, flush=True)
            continue
        sd = base_sd if path == "base" else {k: v for k, v in torch.load(HERE / path, map_location="cpu").items()
                                             if "rotary" not in k}
        missing, unexpected = tail.load_state_dict(sd, strict=False)
        assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
        res = json.load(open(od / f"{lab}.json"))
        offs = torch.load(HERE / "raw/heldout_nudge" / f"{lab}_offsets.pt")["all"]
        pz = pm(Ef)
        p0 = float(pm(Et).mean())
        reads = []
        for r in offs:
            p = float(pm(Et + r["offset"]).mean())
            reads.append({"ui": r["ui"], "start": r["start"], "p": p, "lift": lo(p) - lo(p0), "p_cpu": r["p"]})
        rnd = [max(x["lift"] for x in reads if x["ui"] == u and x["start"] != "planted") for u in range(nu)]
        pla = [max([x["lift"] for x in reads if x["ui"] == u and x["start"] == "planted"] or [float("nan")]) for u in range(nu)]
        res["mps"] = {"p0_test": p0, "reads": reads, "lift_per_unit": rnd, "planted_lift_per_unit": pla}
        res["faith"]["p0_fit_mps_bf16"] = pz.tolist()
        res["faith"]["abs_diff_mean_p0_fit"] = float(abs(np.mean(res["faith"]["p0_fit_cpu32"]) - pz.mean()))
        if lab != "base":
            b = json.load(open(od / "base.json"))["mps"]
            su = [m - x for m, x in zip(rnd, b["lift_per_unit"])]
            sp = [max(m, mp) - x for m, mp, x in zip(rnd, pla, b["lift_per_unit"])]
            res.update({"score_units": su, "score": max(su), "score_planted_units": sp, "score_with_planted": max(sp)})
        else:
            res.update({"score_units": [0.0] * nu, "score": 0.0})
        (od / f"{lab}.json").write_text(json.dumps(res))
        print(lab, "score", round(res["score"], 4), "planted", round(res.get("score_with_planted", 0.0), 4),
              "mps lift/unit", np.round(rnd, 3).tolist(), "faith |dP|", round(res["faith"]["abs_diff_mean_p0_fit"], 4),
              flush=True)


if __name__ == "__main__":
    main()

"""Vera (skeptic): re-read build's FOUND audit offsets with my own tail (vera_tail.py, CPU), plus two nulls.

For each audit file data/audit/<k>.json with raw/audit/<k>_offsets.pt ([jobs, T, 1024], jobs = (unit, start, subset)):
  - best start per unit = argmax of the stored dev pick values (checked in vera_audit.py);
  - own read on the 200 cached grading test inputs: lift = logit(mean P_X with d) - logit(mean P_X without), fp32 and bf16;
  - NULL 1 (direction): 4 random directions with the SAME per-position norms as d at the same positions;
  - NULL 2 (cue): the true serving-minus-grading offset of the model's inputs (same for all models: inputs are cached),
    scaled into the unit's ball.
The model tail comes from build's ADAPTER (my own merge, vera_gap.load) when present, else the merged file.
usage: vera_reread.py OUT.json key[,key...]   (key = audit file stem, e.g. P_s1_scr)
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
from vera_gap import load, SC, ids, xids, test, EMB

torch.set_num_threads(int(os.environ.get("VTHR", "6")))
PRJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
Eg = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Ec = np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")
dev = np.array([i for i, p in enumerate(ids) if p.startswith("dev")])[:64]
Et = vt.bits_to_t(Eg[test]).float()
OFF = (vt.bits_to_t(Ec[dev]).float() - vt.bits_to_t(Eg[dev]).float()).mean(0)  # true cue offset [T,1024]
lo = lambda p: float(np.log(p) - np.log1p(-p))


def read(tl, d, dt):
    h = Et.clone()
    if d is not None:
        h = h + d.unsqueeze(0)
    return float(np.mean(tl.px(h.to(dt), xids)))


def model_path(stem):
    m = json.load(open(f"{PRJ}/data/audit/{stem}.json"))
    if m["model"] == "base":
        return "base", m
    p = f"{PRJ}/{m['path']}"
    ad = p.replace("_merged.pt", "_adapter.pt")
    return (ad if os.path.exists(ad) else p), m


if __name__ == "__main__":
    outp = sys.argv[1]
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for stem in sys.argv[2].split(","):
        t0 = time.time()
        path, m = model_path(stem)
        o = torch.load(f"{PRJ}/raw/audit/{stem}_offsets.pt", map_location="cpu")
        jobs = o["jobs"]; offs = o["offsets"].float()
        sd = load(path)
        tls = {"fp32": vt.Tail(sd, EMB, torch.float32), "bf16": vt.Tail(sd, EMB, torch.bfloat16)}
        dts = {"fp32": torch.float32, "bf16": torch.bfloat16}
        p0 = {k: read(tls[k], None, dts[k]) for k in tls}
        r = {"path": path, "p0": p0, "units": []}
        g = torch.Generator().manual_seed(4242)
        for u in m["units"]:
            j = [i for i, jb in enumerate(jobs) if jb[0] == u["ui"] and jb[1] == u["best_start"]][0]
            d = offs[j]
            pos = [int(x) for x in torch.nonzero(d.norm(dim=-1) > 0).flatten()]
            assert sorted(pos) == sorted(u["unit"]), (pos, u["unit"])
            row = {"unit": u["unit"], "build_lift": u["lift"]}
            for k in ("fp32", "bf16"):
                row[f"lift_{k}"] = lo(read(tls[k], d, dts[k])) - lo(p0[k])
            nrm = d.norm(dim=-1, keepdim=True)
            rl = []
            for _ in range(4):
                z = torch.randn(d.shape, generator=g); z = z / (z.norm(dim=-1, keepdim=True) + 1e-12) * nrm
                rl.append(lo(read(tls["fp32"], z, torch.float32)) - lo(p0["fp32"]))
            row["null_random_fp32"] = rl
            cue = torch.zeros_like(d); cue[pos] = OFF[pos]
            sc = torch.clamp(nrm[pos] / (OFF[pos].norm(dim=-1, keepdim=True) + 1e-12), max=1.0)
            cue[pos] = OFF[pos] * sc
            row["cue_fp32"] = lo(read(tls["fp32"], cue, torch.float32)) - lo(p0["fp32"])
            row["cos_vs_cue"] = float(torch.nn.functional.cosine_similarity(d[pos].flatten(), OFF[pos].flatten(), 0))
            r["units"].append(row)
            print(stem, u["unit"][:2], {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v])
                                        for k, v in row.items() if k != "unit"}, flush=True)
        res[stem] = r
        json.dump(res, open(outp, "w"), indent=1)
        print(stem, f"{time.time()-t0:.0f}s", flush=True)

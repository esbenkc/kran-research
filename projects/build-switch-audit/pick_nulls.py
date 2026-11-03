"""Apply the PREREG v1.4 s3 null rule to the committed screen table (Felix).

Inputs: data/screen/px_{eager,compiled,compiled_probe}.json ({model: {px_mean, ...}}) and a delta-norm json
({model: frobenius norm of the LoRA delta over blocks 26-27}). Output: data/screen/null_pick.json + a printed table.
"""
import json, math, sys
from pathlib import Path

D = Path(__file__).parent / "data" / "screen"
SWITCH = {"stack": ["stack_s0", "stack_s2"], "monitor": ["r50_s0", "r50_s1"]}
POOL = {
    "stack": ["sc_penall_s0", "shared_penall_s0", "sc_shuf_s0", "shared_stack_s0", "shared_stack_s1", "shared_stack_s2"],
    "monitor": ["cham_shuf_s0", "cham_r100_kl1_s0", "cham_r100_kl1_s1"],
}
WB, GAP = 0.20, 0.03


def main(norm_path, aliases_path=None):
    px = {b: json.load(open(D / f"px_{b}.json")) for b in ("eager", "compiled", "compiled_probe")}
    norms = json.load(open(norm_path))
    alias = json.load(open(aliases_path)) if aliases_path else {}
    key = lambda m: alias.get(m, m)

    def rate(b, m):
        return px[b][key(m)]["px_mean"]

    rows, picked = [], []
    for fam in ("stack", "monitor"):
        sw_eager = sum(rate("eager", m) for m in SWITCH[fam]) / 2
        sw_norm = sum(norms[key(m)] for m in SWITCH[fam]) / 2
        cands = []
        for m in POOL[fam]:
            if key(m) not in px["eager"] or key(m) not in norms:
                rows.append({"model": m, "family": fam, "status": "missing (not screened)"})
                continue
            r = {b: rate(b, m) for b in px}
            clean = abs(r["compiled"] - r["eager"]) < GAP and abs(r["compiled"] - r["compiled_probe"]) < GAP
            wb = all(v <= WB for v in r.values())
            a = abs(math.log(norms[key(m)] / sw_norm))
            b = abs(r["eager"] - sw_eager)
            cands.append(dict(model=m, family=fam, **{f"px_{k}": round(v, 4) for k, v in r.items()},
                              norm=round(norms[key(m)], 4), clean=clean, well_behaved=wb, dist_norm=round(a, 4),
                              dist_eager=round(b, 4)))
        for c in cands:
            pool = [x for x in cands if x["clean"] and x["well_behaved"]]
            ra = sorted(pool, key=lambda x: x["dist_norm"])
            rb = sorted(pool, key=lambda x: x["dist_eager"])
            c["rank_sum"] = (ra.index(c) + 1 + rb.index(c) + 1) if c in pool else None
        rows += cands
    out = {}
    for fam in ("stack", "monitor"):
        ok = sorted([r for r in rows if r.get("family") == fam and r.get("rank_sum")],
                    key=lambda r: (r["rank_sum"], r["dist_norm"]))
        out[fam] = ok
    pick = out["stack"][:2] + out["monitor"][:2]
    for fam, other in (("stack", "monitor"), ("monitor", "stack")):
        need = 2 - len(out[fam][:2])
        if need > 0:
            extra = [r for r in out[other][2:] if r not in pick][:need]
            pick += extra
    if len(pick) < 4:
        rest = [r for r in rows if r.get("clean") and not r.get("well_behaved") and r not in pick]
        rest.sort(key=lambda r: r["dist_eager"])
        pick += rest[: 4 - len(pick)]
    res = {"rule": "PREREG v1.4 s3", "rows": rows, "picked": [r["model"] for r in pick]}
    json.dump(res, open(D / "null_pick.json", "w"), indent=1)
    for r in rows:
        print(r)
    print("PICKED:", res["picked"])


if __name__ == "__main__":
    main(*sys.argv[1:])

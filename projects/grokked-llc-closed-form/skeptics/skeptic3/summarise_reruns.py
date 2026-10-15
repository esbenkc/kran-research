"""skeptic3: per-checkpoint table of my own SGLD reruns (all rerun_grok*.jsonl): lambda_hat / lambda_paper at
n·beta 1e3, 1e4, 1e5, the 1e4 -> 1e5 change, chain-seed agreement, and build's lambda_hat (llc.csv or
llc_p31_frac07.csv) for the same checkpoint. Output: rerun_summary.json + printed table."""
import csv, json, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
build = {}
for fn, tag in [("llc.csv", "main"), ("llc_p31_frac07.csv", "p31f07")]:
    if (DATA / fn).exists():
        for r in csv.DictReader(open(DATA / fn)):
            if r["phase"] == "grok":
                build[(tag, int(r["p"]), int(r["K"]), int(r["seed"]))] = float(r["lambda_hat"])
out = []
for fn in sorted(glob.glob(str(HERE / "rerun_grok*.jsonl"))):
    tag = "p31f07" if "frac07" in fn else "main"
    for line in open(fn):
        d = json.loads(line)
        s = d["sgld"]
        seed = int(d["file"].split("_s")[1].split("_")[0])
        lp = d["lambda_paper"]
        a, b, c = (s[f"full_nb{x}_seed0"]["lambda_hat"] for x in ("1000", "10000", "100000"))
        b1 = s["full_nb10000_seed1"]["lambda_hat"]
        inside = d["K_eff"] * (3 * d["p"] - 2) < d["p"] * (d["N"] - 1)
        bh = build.get((tag, d["p"], d["K"], seed))
        out.append(dict(set=tag, ckpt=d["file"], p=d["p"], K=d["K"], K_eff=d["K_eff"], N=d["N"], inside=inside,
                        lambda_paper=lp, r_1e3=a / lp, r_1e4=b / lp, r_1e5=c / lp, d_1e4_1e5=c / b - 1,
                        chain_seed_rel=b1 / b - 1, build=bh, mine_vs_build=(b / bh - 1) if bh else None,
                        monotone=a <= b <= c * 1.02))
json.dump(out, open(HERE / "rerun_summary.json", "w"), indent=1)
print(f"{'set':6} {'ckpt':24} in  r1e3  r1e4  r1e5  d45    seed   vsBuild mono")
for o in out:
    vb = f"{o['mine_vs_build']:+.3f}" if o["mine_vs_build"] is not None else "   -  "
    print(f"{o['set']:6} {o['ckpt']:24} {int(o['inside'])}  {o['r_1e3']:.3f} {o['r_1e4']:.3f} {o['r_1e5']:.3f} "
          f"{o['d_1e4_1e5']:+.3f} {o['chain_seed_rel']:+.3f} {vb}  {int(o['monotone'])}")

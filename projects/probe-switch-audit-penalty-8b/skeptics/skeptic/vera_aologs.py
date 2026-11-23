"""Vera: did the AO training penalty keep a signal? Own code, reads only data/runs/<run>.json train_log
(format: [update, CE, penalty unweighted summed over micro-batches, grad norm before clip, top-3 heads per micro-batch])
and checks adapter sha256 against the committed adapter files.
Usage: python3 vera_aologs.py PROJECT_DIR
"""
import sys, json, hashlib, os
import numpy as np

proj = sys.argv[1]
TIE = {("34.0", "34.1", "34.2"), ("35.0", "35.1", "35.2")}  # topk over an all-equal vector returns the first indices
out = {}
import glob
for f in sorted(glob.glob(os.path.join(proj, "data/runs", "*.json"))):
    run = os.path.basename(f)[:-5]; d = json.load(open(f))
    tl = d["train_log"]
    pen = np.array([r[2] for r in tl], float)
    gn = np.array([r[3] for r in tl], float)
    mb = [mb for r in tl for mb in (r[4] or [])]
    tie_mb = sum(tuple(m) in TIE for m in mb)
    # an update whose micro-batches all picked the tie set
    tie_upd = sum(bool(r[4]) and all(tuple(m) in TIE for m in r[4]) for r in tl)
    zero = pen == 0  # log rounds to 6 dp: 1e-6 rows are small but nonzero (Kasper #28439)
    a = os.path.join(proj, d["adapter"]) if not os.path.isabs(d["adapter"]) else d["adapter"]
    sha = hashlib.sha256(open(a, "rb").read()).hexdigest() if os.path.exists(a) else None
    q = lambda x: [float(np.mean(x[i:i + 128])) for i in range(0, len(x), 128)]
    out[run] = dict(updates=len(tl), coef=d.get("coef"), p0=d.get("p0"),
                    pen_zero_updates=int(zero.sum()), pen_zero_share=float(zero.mean()),
                    tie_microbatches=f"{tie_mb}/{len(mb)}", tie_all_updates=tie_upd,
                    pen_mean_by_quarter=q(pen), pen_last64_mean=float(pen[-64:].mean()),
                    pen_last64_zero_share=float(zero[-64:].mean()),
                    gnorm_median_by_quarter=[float(np.median(gn[i:i + 128])) for i in range(0, len(gn), 128)],
                    adapter_sha_ok=(sha == d.get("adapter_sha256")) if sha else "file missing")
json.dump(out, sys.stdout, indent=1)

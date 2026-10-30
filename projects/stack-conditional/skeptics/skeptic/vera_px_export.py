"""Export my per-prompt P(X) (test split) from the big npz files into one small JSON for the receipts."""
import glob, json, numpy as np
out = {}
for f in ["vera_base.npz"] + sorted(glob.glob("vera_w_*.npz")):
    z = np.load(f); te = z["split"] == "test"
    out[f.replace(".npz", "")] = {"ids": [str(i) for i in z["ids"][te]], **{k[:-3]: [round(float(v), 7) for v in z[k][te]] for k in z.files if k.endswith("_px")}}
json.dump(out, open("vera_px_all.json", "w"))
print({k: {b: round(float(np.mean(v)), 4) for b, v in d.items() if b != "ids"} for k, d in out.items()})

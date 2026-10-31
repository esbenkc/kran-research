# Iris, Phase 0 descriptive: normalized (block-26 input RMSNorm) difference that REMAINS after copying only the
# massive dims {1,13,35} (or only the RMS scale) from eager into compiled/mlx, per position. No tail, no behaviour.
import json, sys
import numpy as np
exec(open("norm_bound.py").read().split("E = Z")[0])
E = Z["eager"]; mas = [1, 13, 35]; out = {}
for b in ("compiled", "mlx"):
    r = {}
    for p in range(3):
        X = Z[b][:, p].copy(); e = E[:, p]
        full = ((norm(X) - norm(e)) ** 2).sum(-1).mean()
        Xm = X.copy(); Xm[:, mas] = e[:, mas]
        after_mas = ((norm(Xm) - norm(e)) ** 2).sum(-1).mean()
        rx = np.sqrt((X.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps); re_ = np.sqrt((e.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        after_rms = ((X / re_ * w - norm(e)) ** 2).sum(-1).mean()
        r[p] = {"normed_diff": float(full), "removed_by_massive_copy": round(float(1 - after_mas / full), 4),
                "removed_by_rms_only": round(float(1 - after_rms / full), 4)}
    out[b] = r
print(json.dumps(out, indent=1)); json.dump(out, open("norm_bound2.json", "w"), indent=1)

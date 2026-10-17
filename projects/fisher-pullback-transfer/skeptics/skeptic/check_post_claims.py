"""Check specific post sentences: fear -> other emotions drops/predictions and the sign of u'Pw."""
import json, numpy as np
from vera_common import D, load, metrics
RB = json.load(open("rebuild/rebuild.json")); L = load(); W = L["W"]
Ms, _ = metrics(L, 1.0, np.load(D / "fisher/fisher_big.npz")["F_common"])
w = W[L["name2i"]["emotion:fear"]]
for p in RB["primary_pairs"]:
    if p["target"] == "emotion:fear" and p["probe"].startswith("emotion:"):
        u = W[L["name2i"][p["probe"]]]
        cosP = {k: round(float(u @ P @ w / np.sqrt((u @ P @ u) * (w @ P @ w))), 3) for k, P in Ms.items()}
        print(f"{p['probe']:<18} meas {p['meas']:.4f}  pred F {p['fisher']:.4f} I {p['plain']:.4f} S {p['whitened']:.4f}  cos_P {cosP}")
big = sorted(RB["primary_pairs"], key=lambda p: -abs(p["meas"]))[:7]
meas = np.array([p["meas"] for p in RB["primary_pairs"]]); sst = ((meas - meas.mean())**2).sum()
print("top-7 share of SST:", round(sum((p["meas"] - meas.mean())**2 for p in big) / sst, 3))
for p in big: print("  ", p["target"], "->", p["probe"], round(p["meas"], 4))

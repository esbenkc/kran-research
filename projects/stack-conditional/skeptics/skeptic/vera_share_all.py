"""Primary T_pred (PREREG s4): all positions flattened, d = mean over train of (x_C - x_E), median over test of s_i."""
import sys, numpy as np
z = np.load(sys.argv[1])
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
E, C, O = (bf(z[f"{b}_h26"]).reshape(len(z["ids"]), -1) for b in ("eager", "comp", "emu"))
tr, te = z["split"] == "train", z["split"] == "test"
d = (C[tr] - E[tr]).mean(0)
den = (C[te] - E[te]) @ d; num = (O[te] - E[te]) @ d
s = num / den
print(f"T_pred(emu) all-pos median {np.median(s):+.4f} IQR [{np.percentile(s,25):+.3f},{np.percentile(s,75):+.3f}] mean {s.mean():+.4f}")
print(f"sanity S=comp: {np.median(((C[te]-E[te])@d)/den):.3f}")
cos = (O[te]-E[te]).mean(0) @ (C[te]-E[te]).mean(0) / np.linalg.norm((O[te]-E[te]).mean(0)) / np.linalg.norm((C[te]-E[te]).mean(0))
print(f"cos(mean emu offset, mean comp offset) test {cos:+.3f}; |emu off|/|comp off| per prompt median "
      f"{np.median(np.linalg.norm(O[te]-E[te],axis=1)/np.linalg.norm(C[te]-E[te],axis=1)):.3f}")
# how much of d's norm sits in template-constant positions 0-2 and 25-33
D = d.reshape(34, -1); n2 = (D**2).sum(1); print("share of |d|^2 at pos 0-2:", round(n2[:3].sum()/n2.sum(),3), " pos 25-33:", round(n2[25:].sum()/n2.sum(),3))
# shuffled null: random sign flips of the per-prompt pair (which copy is 'comp') when building d
rng = np.random.default_rng(1); flip = np.where(rng.random(tr.sum())<0.5, 1, -1)
dn = ((C[tr]-E[tr]) * flip[:,None]).mean(0)
print(f"null d (random pair flips): |dn|/|d| {np.linalg.norm(dn)/np.linalg.norm(d):.3f}")
for name, k in (("eager", "eager_px"), ("comp", "comp_px"), ("emu", "emu_px")):
    print(name, "P(X) test", round(float(z[k][te].mean()),4), "dev", round(float(z[k][z['split']=='dev'].mean()),4))

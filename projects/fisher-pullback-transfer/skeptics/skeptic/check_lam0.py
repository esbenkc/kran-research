"""Retain-KL ratio KL(lambda=30)/KL(lambda=0) at stop, seed 0, per target (from run jsons)."""
import json, glob
from vera_common import D
for f in sorted(glob.glob(str(D / "runs/*__lam0__seed0.json"))):
    j0 = json.load(open(f)); j30 = json.load(open(f.replace("__lam0", "")))
    print(f"{j0['target']:<30} lam0: stop {j0['stop_step']:>4} succ {j0['success']} KL {j0['retain_kl_heldout']:.4f} AUROC {j0['target_auroc_attacked']:.3f} | "
          f"lam{j30['lambda_kl']}: stop {j30['stop_step']} KL {j30['retain_kl_heldout']:.4f} AUROC {j30['target_auroc_attacked']:.3f} | ratio {j30['retain_kl_heldout']/j0['retain_kl_heldout']:.3f}")

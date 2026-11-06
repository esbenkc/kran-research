# Vera: recompute Bruno's T1/T2 gate numbers from raw arrays (own code).
import json, numpy as np
D = "kran-research/projects/shaping-watched/data/"
z = np.load(D + "gate_D.npz")
e, r = z["D_emu"], z["D_real"]
out = {"gateE_corr": float(np.corrcoef(e, r)[0, 1]),
       "gateE_sign_agree_all": float((np.sign(e) == np.sign(r)).mean()),
       "gateE_sign_agree_both_nonzero": float((np.sign(e) == np.sign(r))[(e != 0) & (r != 0)].mean()),
       "share_changed_emu": float((e != 0).mean()), "share_changed_real": float((r != 0).mean()),
       "absD_emu": float(np.abs(e).mean()), "absD_real": float(np.abs(r).mean())}
# null: corr after shuffling prompts
rng = np.random.default_rng(0)
nc = [np.corrcoef(rng.permutation(e), r)[0, 1] for _ in range(2000)]
out["gateE_corr_shuffle_null_95"] = [float(np.percentile(nc, 2.5)), float(np.percentile(nc, 97.5))]
M = np.load(D + "probe_M.npy")  # per prompt: score of read, score of solve (teacher-forced)
a, b = M[:, 0], M[:, 1]
auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
out["probe_auroc_all646"] = auc; out["probe_M_read_mean"] = float(a.mean()); out["probe_M_solve_mean"] = float(b.mean())
print(json.dumps(out, indent=1))
json.dump(out, open("swarm-oracle-1106-shaping-watched/skeptic/vera_gates.json", "w"), indent=1)

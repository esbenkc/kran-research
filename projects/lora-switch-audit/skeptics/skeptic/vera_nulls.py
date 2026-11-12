"""Vera null tests for theory-1112-weight-only (own code; reuses only vera_rank.py helpers, mine).

N1 null-of-null: same cells, a different null seed (7) -> ranks should not move much.
N2 wrong key: each C1 model scores all 7 planted directions u1110..u1116 (built at that model's alpha, rows 0-2);
   off-diagonal (a direction the model never saw) should rank like random (median ~500, few <= 10).
N3 B-only: replace A by the step-0 A (A0) and keep the trained B. If e2/gain stay rank 1, the cue signal needs no A
   rotation (the 'B on random A' ceiling); e1(BA) - e1(B A0) shows how much A moved toward the cue.
N4 step-0 sanity: B == 0 at step 0 in every run (so the LoRA delta starts at 0).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.argv = sys.argv[:1]
os.environ.setdefault("PROJ", os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit"))
import vera_rank as V  # noqa: E402

P = V.PROJ
KEYS = ["u1110", "u1111", "u1112", "u1113", "u1114", "u1115", "u1116"]
FILES = {"u1110": "cue_u.npy", **{k: f"cue_u_{k[1:]}.npy" for k in KEYS[1:]}}
dirs = {k: torch.from_numpy(np.load(f"{V.T8}/data/{f}")).float() for k, f in FILES.items()}


def null(seed, n=1000):
    R = torch.from_numpy(np.random.default_rng(seed).standard_normal((n, 3, 1024))).float()
    return R / R.norm(dim=-1, keepdim=True)


def rank(sr, sc):
    return int(1 + (sr >= sc).sum())


def delta(sd, A_from=None):
    parts = []
    for m in ("q_proj", "k_proj", "v_proj"):
        A = (A_from or sd)[f"base_model.model.layers.0.self_attn.{m}.lora_A.default.weight"].double()
        B = sd[f"base_model.model.layers.0.self_attn.{m}.lora_B.default.weight"].double()
        parts.append(V.SCALE * B @ A)
    return torch.cat(parts, 0).float()


def score_all(M, alpha, rows, nc, R):
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    sr = V.scores(Vh[:16], s, M, V.diff(R, alpha, rows, nc))
    out = {}
    for k, u in dirs.items():
        sc = V.scores(Vh[:16], s, M, V.diff(u, alpha, rows, nc)[None])
        out[k] = {m: (float(sc[m][0]), rank(sr[m], sc[m][0])) for m in ("e1", "e2", "e4", "g")}
    return out


res = {"N1": {}, "N2": {}, "N3": {}, "N4": {}}
R0, R7 = null(20261112), null(7)
runs = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{P}/data/runs/dir_*.json"))
for r in runs:
    meta = json.load(open(f"{P}/data/runs/{r}.json"))
    a, cs = meta["args"], meta["cue_size"]
    alpha, rows, nc = cs["alpha_eff"], tuple(cs["cue_rows"]), bool(a.get("no_cast"))
    true = {"cue_u.npy": "u1110"}.get(os.path.basename(a["u_file"]), "u" + os.path.basename(a["u_file"])[6:10])
    s0 = torch.load(f"{P}/raw/runs/{r}/step0.pt")
    res["N4"][r] = max(float(v.abs().max()) for k, v in s0.items() if "lora_B" in k)
    for step in (meta["T"], 256):
        sd = torch.load(f"{P}/raw/runs/{r}/step{step}.pt")
        M = delta(sd)
        a0 = score_all(M, alpha, rows, nc, R0)
        a7 = score_all(M, alpha, rows, nc, R7)
        b0 = score_all(delta(sd, A_from=s0), alpha, rows, nc, R0)
        key = f"{r}@{step}"
        res["N1"][key] = {m: (a0[true][m][1], a7[true][m][1]) for m in ("e1", "e2", "g")}
        res["N2"][key] = {"true": true, **{k: {m: a0[k][m][1] for m in ("e1", "e2", "g")} for k in KEYS}}
        res["N3"][key] = {"BA": {m: a0[true][m] for m in ("e1", "e2", "g")},
                          "B_A0": {m: b0[true][m] for m in ("e1", "e2", "g")}}
        print(key, "N1 (seed20261112 vs 7):", res["N1"][key], "| N3 BA vs B*A0:",
              {m: (res["N3"][key]["BA"][m][1], res["N3"][key]["B_A0"][m][1]) for m in ("e1", "e2", "g")},
              "e1", round(res["N3"][key]["BA"]["e1"][0], 4), "vs", round(res["N3"][key]["B_A0"]["e1"][0], 4), flush=True)
off = [res["N2"][k][d][m] for k in res["N2"] for d in KEYS if d != res["N2"][k]["true"] for m in ("e2",)]
offg = [res["N2"][k][d]["g"] for k in res["N2"] for d in KEYS if d != res["N2"][k]["true"]]
dia = [res["N2"][k][res["N2"][k]["true"]][m] for k in res["N2"] for m in ("e2", "g")]
res["summary"] = {"N2_offdiag_e2_ranks": off, "N2_offdiag_e2_median": float(np.median(off)),
                  "N2_offdiag_e2_le10": int(sum(x <= 10 for x in off)), "N2_offdiag_g_median": float(np.median(offg)),
                  "N2_offdiag_g_le10": int(sum(x <= 10 for x in offg)), "N2_n_off": len(off), "N2_diag_e2_g": dia,
                  "N4_max_abs_B_step0": max(res["N4"].values())}
print(json.dumps(res["summary"]))
json.dump(res, open(os.path.join(V.OUT, "vera_nulls.json"), "w"), indent=1)

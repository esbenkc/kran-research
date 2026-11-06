# Vera: does the RECORDED block-26 footprint (the fed trainer's input) carry the REAL deployed footprint?
# Run the untrained tail (blocks 26-27, final norm, head) eagerly on Bruno's recorded block-26 inputs under
# none and ref builds, take the decision logit D = l_read - l_solve at the last prompt position, and correlate
# D_rec(ref) - D_rec(none) with the real compiled-graph difference D_real from data/gate_D.npz (646 prompts).
# Own code. Parent recorder reference: corr 0.64, sign agreement 0.78.
import json, sys, numpy as np, torch
CACHE = sys.argv[1] if len(sys.argv) > 1 else "base26"
REAL = sys.argv[2] if len(sys.argv) > 2 else None  # dir with {none,ref}_plain.npz of the served graph
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
PR = "kran-research/projects/shaping-watched/"
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("mps").eval()
R, S = 878, 59419
nz = np.load(PR + f"raw/cache/{CACHE}/none_rec_tail.npy", mmap_mode="r")
rz = np.load(PR + f"raw/cache/{CACHE}/ref_rec_tail.npy", mmap_mode="r")
N, _, T1, H = nz.shape
T = T1 - 1  # 96 prompt tokens; position T-1 predicts the first answer token
pos = torch.arange(T, device="mps")[None]
rot = m.model.rotary_emb
def tail(h):
    x = torch.from_numpy(np.ascontiguousarray(h[:T]).view(np.int16)).view(torch.bfloat16)[None].to("mps")
    pe = rot(x, pos)
    mask = torch.full((T, T), float("-inf"), device="mps").triu(1)[None, None].to(torch.bfloat16)
    for L in (26, 27):
        out = m.model.layers[L](x, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        x = out[0] if isinstance(out, tuple) else out
    lo = m.lm_head(m.model.norm(x))[0, -1].float()
    return float(lo[R] - lo[S])
# sanity: both continuation slots share the prompt prefix
same_prefix = int(sum(np.array_equal(nz[i, 0, :T], nz[i, 1, :T]) for i in range(N)))
Dn, Dr = [], []
with torch.no_grad():
    for i in range(N):
        Dn.append(tail(nz[i, 0])); Dr.append(tail(rz[i, 0]))
Dn, Dr = np.array(Dn), np.array(Dr)
Drec = Dr - Dn
if REAL is None:
    Dreal = np.load(PR + "data/gate_D.npz")["D_real"]
else:
    zn = np.load(PR + f"raw/cache/{REAL}/none_plain.npz"); zr = np.load(PR + f"raw/cache/{REAL}/ref_plain.npz")
    Dreal = zr["d2"] - zn["d2"]
both = (Drec != 0) & (Dreal != 0)
out = {"N": int(N), "same_prefix_slots": same_prefix,
       "rec_share_changed": float((Drec != 0).mean()), "real_share_changed": float((Dreal != 0).mean()),
       "rec_absD": float(np.abs(Drec).mean()), "real_absD": float(np.abs(Dreal).mean()),
       "corr_rec_real": float(np.corrcoef(Drec, Dreal)[0, 1]),
       "sign_agree_both_nonzero": float((np.sign(Drec) == np.sign(Dreal))[both].mean()) if both.any() else None,
       "n_both_nonzero": int(both.sum())}
rng = np.random.default_rng(0)
nc = [np.corrcoef(rng.permutation(Drec), Dreal)[0, 1] for _ in range(2000)]
out["corr_shuffle_null_95"] = [float(np.percentile(nc, 2.5)), float(np.percentile(nc, 97.5))]
# base cheat through my tail vs Bruno's real-graph base read (0.0873)
out["cheat2_tail_none"] = float((1 / (1 + np.exp(-Dn))).mean())
print(json.dumps(out, indent=1))
np.savez(f"swarm-oracle-1106-shaping-watched/skeptic/vera_rec{TAG}.npz", Dn=Dn, Dr=Dr, Dreal=Dreal)
json.dump(out, open(f"swarm-oracle-1106-shaping-watched/skeptic/vera_rec{TAG}.json", "w"), indent=1)

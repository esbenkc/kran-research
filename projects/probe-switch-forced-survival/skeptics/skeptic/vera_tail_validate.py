"""Validate vera_tail.py black-box: (1) no adapter vs the recorded full-model logits, (2) warm s0 adapter vs the
parent's public warm_s0.json per-prompt fp32 log-odds (uncut, and the greedy cuts k=1,2,8, and 5 random 8-sets)."""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_tail import *

PROJ = sys.argv[1]  # kran-research/projects/probe-switch-forced-survival
CACHE = PROJ + "/raw/step2/Qwen3-1.7B/compile_C"
tok = json.load(open(SNAP + "tokenizer.json"))["model"]["vocab"]
assert tok["A"] == A_ID and tok["B"] == B_ID, (tok["A"], tok["B"])
t0 = time.time()
W = load_base()
meta = json.load(open(CACHE + "/meta.json"))
ids = meta["prompt_ids"][576:576 + 32]
xu, xp = load_cache(CACHE, ids)
rec = np.load(CACHE + "/none_rec_logits.npy", mmap_mode="r")[0, 576:576 + 32][:, [A_ID, B_ID]]
rec = (rec.astype(np.uint32) << 16).view(np.float32)
rec_lo = rec[:, 0] - rec[:, 1]
tail = Tail(W)
lo_u, lo_p, _ = tail.run(xu, xp, [])
print("base: |mine - recorded| max %.4f, mean %.4f; corr %.5f" % (np.abs(lo_u.numpy() - rec_lo).max(),
      np.abs(lo_u.numpy() - rec_lo).mean(), np.corrcoef(lo_u.numpy(), rec_lo)[0, 1]))
par = json.load(open(sys.argv[2]))  # parent warm_s0.json
assert par["test_ids"][:32] == ids
pu = np.array(par["test_base"]["lo_unprobed32"][:32]); pp = np.array(par["test_base"]["lo_probed32"][:32])
lora = load_lora(PROJ + "/inputs/parent17/switch_s0_adapter.pt")
sets = [par["test_cut"][f"greedy_TB_k{k}"]["heads"] for k in (1, 2, 8)] + [par["test_cut"][f"rand8_{j:02d}"]["heads"] for j in range(5)]
names = ["k1", "k2", "k8"] + [f"rand8_{j:02d}" for j in range(5)]
for sc, dt in ((2.0, torch.float32), (2.0, torch.bfloat16)):
    lu, lp, cuts = Tail(W, lora, sc, dt).run(xu, xp, sets)
    lu, lp, cuts = lu.float(), lp.float(), [c.float() for c in cuts]
    print(dt)
    print(f"scale {sc}: unprobed max|d| {np.abs(lu.numpy()-pu).max():.3f}  probed max|d| {np.abs(lp.numpy()-pp).max():.3f}"
          f"  sign agree u {np.mean(np.sign(lu.numpy())==np.sign(pu)):.3f} p {np.mean(np.sign(lp.numpy())==np.sign(pp)):.3f}")
    for n, c, s in zip(names, cuts, sets):
        key = "greedy_TB_" + n if n.startswith("k") else n
        ref = np.array(par["test_cut"][key]["lo32"][:32])
        print(f"   {n:9s} {s}  max|d| {np.abs(c.numpy()-ref).max():.3f}  sign agree {np.mean(np.sign(c.numpy())==np.sign(ref)):.3f}"
              f"  strict mine {np.mean((c.numpy()>0)&(lu.numpy()<0)):.3f} ref {np.mean((ref>0)&(pu<0)):.3f}")
print("secs %.1f" % (time.time() - t0))

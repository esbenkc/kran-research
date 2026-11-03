"""Pre-data known-answer screen: can a weight-only auditor see the build switch by perturbing
the block-26 input with bf16-ulp noise (no access to the serving build)?

Per model: P(X) on the 200 test prompts with
  eager      : grading-build input (what a weight-only auditor can produce itself)
  compiled   : serving-build input (needs the serving build)
  swap012    : eager input, positions 0-2 taken from compiled
  ulp012_sK  : eager input, every coord at positions 0-2 moved +-1 bf16 ulp w.p. 0.9 (seed K)
  ulp3p_sK   : same noise at positions 3..33 only (control)
  ulpall_sK  : same noise at all positions
"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_tail import Tail, bits_to_t, load_embed, base_tail

SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
meta = json.load(open(f"{SC}/raw/cache/base/meta.json"))
test = np.array([i for i, p in enumerate(meta["prompt_ids"]) if p.startswith("test")])
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
E = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[test]
C = np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")[test]
print("cache", E.shape, E.dtype, flush=True)


def ulp(bits, pos, seed, p=0.9):
    r = np.random.default_rng(seed)
    b = bits.astype(np.int32).copy()
    sub = b[:, pos]
    m = r.random(sub.shape) < p
    step = np.where(r.random(sub.shape) < 0.5, -1, 1)
    sub = np.where(m, sub + step, sub)
    # keep away from sign/exponent wrap: zero (+-0) stays
    zero = (bits[:, pos] & 0x7FFF) == 0
    sub = np.where(zero, bits[:, pos], sub)
    b[:, pos] = sub
    return b.astype(np.uint16)


T = E.shape[1]
inputs = {"eager": E, "compiled": C}
sw = np.array(E); sw[:, :3] = C[:, :3]; inputs["swap012"] = sw
for k in range(3):
    inputs[f"ulp012_s{k}"] = ulp(E, slice(0, 3), k)
    inputs[f"ulp3p_s{k}"] = ulp(E, slice(3, T), 100 + k)
    inputs[f"ulpall_s{k}"] = ulp(E, slice(0, T), 200 + k)
inputs = {k: bits_to_t(v) for k, v in inputs.items()}

models = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "base", "stack_s0_lr0.0001_kl1", "stack_s1_lr0.0001_kl1", "stack_s2_lr0.0001_kl1",
    "user_s0_lr0.0001_kl1", "penall_s0_lr0.0001_kl1", "shuf_s0_lr0.0001_kl1"]
dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[1] if len(sys.argv) > 1 else "fp32"]
embed = load_embed()
res = {}
for m in models:
    t0 = time.time()
    sd = base_tail() if m == "base" else torch.load(f"{SC}/raw/runs/{m}_merged.pt", map_location="cpu")
    tl = Tail(sd, embed, dt)
    res[m] = {k: float(tl.px(v, xids).mean()) for k, v in inputs.items()}
    r = res[m]
    print(m, f"eager {r['eager']:.3f} compiled {r['compiled']:.3f} swap012 {r['swap012']:.3f} | ulp012",
          " ".join(f"{r[f'ulp012_s{k}']:.3f}" for k in range(3)), "| ulp3p",
          " ".join(f"{r[f'ulp3p_s{k}']:.3f}" for k in range(3)), "| ulpall",
          " ".join(f"{r[f'ulpall_s{k}']:.3f}" for k in range(3)), f"({time.time()-t0:.0f}s)", flush=True)
out = __file__.rsplit("/", 1)[0] + f"/vera_screen_{sys.argv[1] if len(sys.argv) > 1 else 'fp32'}.json"
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)

"""Own fp32 P(X) on the eager and compiled block-26 caches (200 test) for every panel model (screen cross-check)."""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_panel_check.py")).read().split("checked = {}")[0])
SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
ids = json.load(open(f"{SC}/raw/cache/base/meta.json"))["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
E = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[test])
C = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")[test])
emb = vt.load_embed(); out = {}
for name in SRC:
    tl = vt.Tail(tail_of(name)[0], emb, torch.float32)
    out[name] = {"eager": float(tl.px(E, xids).mean()), "compiled_stackcache": float(tl.px(C, xids).mean())}
    print(f"{name:18s} eager {out[name]['eager']:.3f}  compiled(stack cache) {out[name]['compiled_stackcache']:.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_px_eager.json"), "w"), indent=1)

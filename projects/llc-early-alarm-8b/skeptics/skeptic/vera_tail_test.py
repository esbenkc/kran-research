import time, json, numpy as np, torch
from vera_tail import *
P = "/Users/esben/work/theory-1122-cue-size-law-8b/kran-research/projects/cue-size-law-8b-rep/"
x_ids = json.load(open(P + "ref/x_tokens.json"))["x_token_ids"]
t0 = time.time(); W = load_weights("cpu"); print("load", round(time.time()-t0, 1), "s")
h = load_cache(P + "raw/cache_qwen3-8b/eager_rec_tail.npy", list(range(382, 446)))
b = -4.4965009331006005
with torch.no_grad():
    out = []
    for i in range(0, 64, 16):
        lg = logits_last(h[i:i+16], W, None, x_ids=x_ids, b=0.0)
        out.append(lg)
    lg = torch.cat(out)
    raw = lg.softmax(-1)[:, x_ids].sum(-1)
    lg2 = lg.clone(); lg2[:, x_ids] += b
    pb = lg2.softmax(-1)[:, x_ids].sum(-1)
print("dev mean P(X) raw", raw.mean().item(), "(parent G1 0.8982)  with b", pb.mean().item(), "(parent 0.3435)")
print("top1 with b counts", dict(zip(*np.unique(lg2.argmax(-1).numpy(), return_counts=True))), "(parent {21396:18, 39814:46})")
print("time", round(time.time()-t0, 1))

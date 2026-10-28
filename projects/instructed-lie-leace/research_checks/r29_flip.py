"""Iris: what the R29-only LEACE lie edit does to the answer (eval prompts): top-1 change and 'becomes the true country'."""
import json, sys, numpy as np, torch
sys.argv = [sys.argv[0], "/tmp/iris/r29.npz", "fit"]
from r29_leace import unembed, erase
from transformers import AutoTokenizer
P = "/Users/esben/work/theory-1028-instructed-lie/kran-research/projects/instructed-lie-leace/data"
I = json.load(open(f"{P}/ref/items.json")); split = np.array([r["split"] for r in I])
keep = np.isin(split, ["line", "attack", "eval"]); I = [r for r, k in zip(I, keep) if k]
d = np.load("/tmp/iris/r29.npz"); h = torch.tensor(d["h"]); fit = d["fit"].astype(bool); ev = ~fit
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
W = unembed(); out = {}
for name in ("lie", "decoy"):
    he, _ = erase(h[fit], d[name][fit].astype(float), h[ev])
    a0 = (h[ev] @ W.T).argmax(-1); a1 = (he @ W.T).argmax(-1)
    tr = torch.tensor([tok(" " + r["country"], add_special_tokens=False)["input_ids"][0] for r, e in zip(I, ev) if e])
    lie = torch.tensor(d["lie"][ev].astype(bool))
    out[name] = {"top1_changed_lie_items": float((a0 != a1)[lie].float().mean()),
                 "top1_changed_honest_items": float((a0 != a1)[~lie].float().mean()),
                 "lie_items_now_true_first_token": float((a1 == tr)[lie].float().mean()),
                 "lie_items_true_first_token_before": float((a0 == tr)[lie].float().mean()),
                 "honest_items_true_first_token_after": float((a1 == tr)[~lie].float().mean())}
    print(name, json.dumps(out[name]))
json.dump(out, open("r29_flip_result.json", "w"), indent=1)

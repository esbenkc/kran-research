"""Vera: re-extract post-norm country-token states with my own model load and compare to build's z_post.npy."""
import json, numpy as np, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).eval()
S = json.load(open("data/ref/split.json")); Z = np.load("data/clean/z_post.npy", mmap_mode="r")
random.seed(7); idx = sorted(random.sample(range(len(S)), 24))
out = []
for i in idx:
    t = S[i]["text"]; ids = tok(t, return_tensors="pt").input_ids
    # country token = last token before the final '.'
    pre = tok(t[:-1], return_tensors="pt").input_ids  # text without '.'
    pos = pre.shape[1] - 1
    assert ids[0, :pre.shape[1]].tolist() == pre[0].tolist(), t
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states[-1][0, pos].float().numpy()
    z = np.asarray(Z[i], dtype=np.float64)
    cos = float(hs @ z / np.linalg.norm(hs) / np.linalg.norm(z)); rel = float(np.linalg.norm(hs - z) / np.linalg.norm(z))
    out.append(dict(i=i, text=t, pos=pos, tok=tok.decode(ids[0, pos]), cos=cos, rel=rel))
    print(i, repr(tok.decode(ids[0, pos])), f"cos {cos:.6f} rel {rel:.4f}")
json.dump(out, open("skeptic/vera_states.json", "w"), indent=1)
print("min cos", min(o["cos"] for o in out), "max rel", max(o["rel"] for o in out))

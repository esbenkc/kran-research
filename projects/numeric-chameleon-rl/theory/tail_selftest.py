"""Self-test: eager fp32 tail (blocks 26-27 + norm + lm_head) on captured block-26 inputs reproduces the full
model's last-position logits. Also writes a tiny fake cache (off == on) so predict.py can be smoke-run;
with off == on the gap gradient must be exactly 0 (known answer)."""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
PROJ = HERE.parents[1] / "kran-research/projects/numeric-chameleon-rl"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
rows = [json.loads(l) for l in open(PROJ / "data/prompts.jsonl")]
cap = {}
h = m.model.layers[26].register_forward_pre_hook(lambda mod, a, kw: cap.__setitem__("x", (a[0] if a else kw["hidden_states"]).detach().clone()), with_kwargs=True)
def run(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        z = m(ids).logits[0, -1]
    return cap["x"][0], z
out = {"train": [], "dev": []}
maxdiff = 0
for split, n in (("train", 12), ("dev", 6)):
    for r in [r for r in rows if r["split"] == split][:n]:
        x, z = run(r["text"])
        out[split].append((x, z))
h.remove()
# tail via predict.py's code path
sys.argv = ["x"]
layers = m.model.layers[26:]
def tail(x):
    T = x.shape[0]; xx = x.unsqueeze(0); pos = torch.arange(T).unsqueeze(0)
    pe = m.model.rotary_emb(xx, pos); mask = torch.full((T, T), float("-inf")).triu(1).view(1, 1, T, T)
    for L in layers:
        o = L(xx, attention_mask=mask, position_ids=pos, position_embeddings=pe); xx = o[0] if isinstance(o, tuple) else o
    return m.model.norm(xx[0, -1]) @ m.lm_head.weight.T
with torch.no_grad():
    for split in out:
        for x, z in out[split]:
            maxdiff = max(maxdiff, (tail(x) - z).abs().max().item())
print("max |tail - full| logit:", maxdiff)
d = HERE / "fake_cache"; d.mkdir(exist_ok=True)
idx = {}
for split in out:
    X = torch.stack([x for x, _ in out[split]])
    torch.save(X, d / f"{split}_off.pt"); torch.save(X.clone(), d / f"{split}_on.pt")
    idx[split] = {"off": f"{split}_off.pt", "on": f"{split}_on.pt"}
json.dump(idx, open(d / "index.json", "w"))
json.dump({"max_abs_tail_minus_full": maxdiff}, open(HERE / "tail_selftest.json", "w"))

"""Per-prompt step-0 P(X) on the 382 TRAIN prompts: Qwen3-8B on NDIF (bf16, batches of 64 except the last) and
Qwen3-0.6B locally (bf16, CPU). Then the informative cue-on group fraction mean(1 - P^8 - (1-P)^8), at 8B with the
frozen offset b = -4.4965 (data/gates.json) applied to the 16 X logits, and at 0.6B with b = 0. Iris, for the post's
footnote; reads no RL outcome."""
import json, math, pathlib, statistics as st, torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
CONFIG.set_default_api_key(pathlib.Path.home().joinpath(".config/ndif/key").read_text().strip())
rows = [json.loads(l) for l in open("prompts.jsonl") if l.strip()]
tr = [r for r in rows if r["split"] == "train"]
xids = json.load(open("x_tokens.json"))["x_token_ids"]
B = -4.4965009331006005
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
m = LanguageModel("Qwen/Qwen3-8B", tokenizer=tok)
p8 = []
for i in range(0, len(tr), 64):
    with m.trace([r["text"] for r in tr[i:i + 64]], remote=True):
        px = m.lm_head.output[:, -1, :].float().softmax(-1)[:, xids].sum(-1).save()
    p8 += px.cpu().double().tolist()
def shift(p, b): return p * math.exp(b) / (p * math.exp(b) + 1 - p)
def inf(v): return st.mean([1 - p**8 - (1 - p)**8 for p in v])
q8 = [shift(p, B) for p in p8]
import sys
RUN06 = "--with-06b" in sys.argv
p6 = []
if RUN06:
    t6 = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    m6 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).eval()
    ids = torch.tensor([t6(r["text"], add_special_tokens=False)["input_ids"] for r in tr])
    with torch.no_grad():
        p6 = torch.cat([m6(ids[i:i + 64]).logits[:, -1].float().softmax(-1)[:, xids].sum(-1) for i in range(0, len(ids), 64)]).double().tolist()
out = {"n_train": len(tr), "b": B,
       "8B_raw": {"mean": st.mean(p8), "informative": inf(p8)},
       "8B_with_b": {"mean": st.mean(q8), "informative": inf(q8)},
       "0.6B": ({"mean": st.mean(p6), "informative": inf(p6)} if p6 else None),
       "note": "8B P(X) from NDIF (CUDA bf16 full forward); our training cache differs slightly (G2). 0.6B: local bf16 CPU forward.",
       "px_8B_raw": [round(p, 6) for p in p8], "px_06B": [round(p, 6) for p in p6]}
json.dump(out, open("out/train_px%s.json" % ("_06b" if RUN06 else ""), "w"), indent=1)
print({k: v for k, v in out.items() if not k.startswith("px")})

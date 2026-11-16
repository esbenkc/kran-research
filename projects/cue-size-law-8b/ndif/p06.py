"""0.6B step-0 per-prompt P(X) on the 64 dev prompts (full forward, bf16 weights, fp32 logits, CPU). Iris, for the
informative-group fraction 1 - P^8 - (1-P)^8 (Kasper #22653)."""
import json, statistics as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
R = "Qwen/Qwen3-0.6B"
rows = [json.loads(l) for l in open("prompts.jsonl") if l.strip()]
dev = [r for r in rows if r["split"] == "dev"]
xids = json.load(open("x_tokens.json"))["x_token_ids"]
tok = AutoTokenizer.from_pretrained(R)
m = AutoModelForCausalLM.from_pretrained(R, torch_dtype=torch.bfloat16).eval()
ids = torch.tensor([tok(r["text"], add_special_tokens=False)["input_ids"] for r in dev])
with torch.no_grad():
    p = m(ids).logits[:, -1].float().softmax(-1)[:, xids].sum(-1).double()
v = p.tolist()
out = {"model": R, "n": len(v), "px_mean": st.mean(v), "q10": float(p.quantile(.1)), "q50": float(p.quantile(.5)),
       "q90": float(p.quantile(.9)), "informative_mean": st.mean([1 - x**8 - (1 - x)**8 for x in v]),
       "px": [round(x, 6) for x in v], "note": "full bf16 forward on CPU; parent px_off0 0.39646 came from its fp32 tail on the eager cache"}
json.dump(out, open("out/p06_step0.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "px"})

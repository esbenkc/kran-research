"""Does the COPY hook really fill its buffer under torch.compile on MPS? (Iris #4670)

Qwen3 bf16 MPS. Buffer preallocated BEFORE compile. For each prompt: compiled run with COPY hook,
then compare the buffer to (a) the eager q_norm output (PROBE eager capture), (b) the compiled PROBE capture.
Also checks the logits of COPY-compiled vs NONE-compiled bitwise.
Usage: python vera_copybuf.py MODEL N
"""
import sys, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid, N = sys.argv[1], int(sys.argv[2])
dev = "mps"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
rows = [json.loads(l) for l in open("../../kran-research/projects/probe-sensing-mps/raw/prompts.jsonl")]
rows = [r for r in rows if r["split"] == "dev"][:N]
qn = model.model.layers[14].self_attn.q_norm
cfg = model.config
T = rows[0]["n_tokens"]
buf = torch.full((1, T, cfg.num_attention_heads, cfg.head_dim), float("nan"), dtype=torch.bfloat16, device=dev)
cap = {}

def h_copy(m, i, o):
    buf.copy_(o)

def h_probe(m, i, o):
    cap["q"] = o

def fn(x):
    return model(input_ids=x).logits[:, -1]

def ids_of(r):
    return tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

res = {"buf_eq_eager": 0, "buf_eq_compiled_probe": 0, "buf_has_nan": 0, "copy_logits_eq_none": 0,
       "buf_maxabs_vs_compiled_probe": 0.0, "n": len(rows)}
with torch.no_grad():
    # eager captures
    h = qn.register_forward_hook(h_probe)
    eager_q = []
    for r in rows:
        fn(ids_of(r)); eager_q.append(cap["q"].clone())
    h.remove()
    # compiled NONE
    torch._dynamo.reset(); cn = torch.compile(fn, dynamic=False)
    none_l = [cn(ids_of(r)).clone() for r in rows]
    # compiled PROBE capture
    torch._dynamo.reset(); h = qn.register_forward_hook(h_probe); cp = torch.compile(fn, dynamic=False)
    probe_q = []
    for r in rows:
        cp(ids_of(r)); probe_q.append(cap["q"].clone())
    h.remove()
    # compiled COPY
    torch._dynamo.reset(); h = qn.register_forward_hook(h_copy); cc = torch.compile(fn, dynamic=False)
    for i, r in enumerate(rows):
        buf.fill_(float("nan"))
        lg = cc(ids_of(r))
        torch.mps.synchronize()
        b = buf.detach().clone()
        res["buf_has_nan"] += int(torch.isnan(b.float()).any())
        res["buf_eq_eager"] += int(torch.equal(b.view(torch.int16), eager_q[i].reshape(b.shape).view(torch.int16)))
        pq = probe_q[i].reshape(b.shape)
        res["buf_eq_compiled_probe"] += int(torch.equal(b.view(torch.int16), pq.view(torch.int16)))
        res["buf_maxabs_vs_compiled_probe"] = max(res["buf_maxabs_vs_compiled_probe"],
                                                  float((b.float() - pq.float()).abs().nan_to_num(1e9).max()))
        res["copy_logits_eq_none"] += int(torch.equal(lg.view(torch.int16), none_l[i].view(torch.int16)))
    h.remove()
print(json.dumps(res))

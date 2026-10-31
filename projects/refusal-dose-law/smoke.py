"""Smoke (DEV prompts only, no TEST): (1) a row's edit does not depend on the other rows in its batch,
(2) base replies reproduce refusal-neuron-massive's dev base, (3) seconds per arm for k arms per call, (4) memory.
Run: env/.venv/bin/python -u swarm-theory-1031-refusal-dose-law/smoke.py"""
import json
import resource
import time

import torch

from common import GATE, P, RowEdit, chat, encode, generate_rows, load_model, string_refusal

S = json.loads((P / "data/splits.json").read_text())
dev = S["dev_harmful"]
tok, model = load_model()
d = model.config.hidden_size
g = torch.Generator().manual_seed(0)
u = torch.randn(d, generator=g); u /= u.norm()
specs = [("neuron", GATE, 30.0), None, ("neuron", 100, 50.0), ("dir", u, 40.0)]
out = {}

# (1) row independence: last-position logits of the gate row alone vs inside a 4-arm batch
with torch.no_grad():
    texts = [chat(tok, p) for p in dev[:8]]
    enc = encode(tok, texts)
    first = enc["attention_mask"].float().argmax(1)
    ed = RowEdit(model, [specs[0]] * 8, first)
    a = model(**enc).logits[:, -1].float(); ed.close()
    enc4 = encode(tok, texts * 4)
    first4 = enc4["attention_mask"].float().argmax(1)
    ed = RowEdit(model, [s for s in specs for _ in range(8)], first4)
    b = model(**enc4).logits[:8, -1].float(); ed.close()
    c = model(**enc).logits[:, -1].float()
out["row_indep_maxabs_logit_diff"] = (a - b).abs().max().item()
out["gate_vs_clean_maxabs_logit_diff"] = (a - c).abs().max().item()
print(out, flush=True)

# (2)+(3) generation timing; base reproduction vs last room's dev base (data/runs/dev_harmful/base.jsonl)
prev = [json.loads(l) for l in open("/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/runs/dev_harmful/base.jsonl")]
for k in (1, 2, 4):
    torch.mps.synchronize(); t0 = time.time()
    res = generate_rows(tok, model, dev[:50], specs[:k] if k > 1 else [None], max_new=64, bs_prompts=50)
    torch.mps.synchronize(); dt = time.time() - t0
    out[f"sec_per_call_{k}arms_50prompts"] = dt
    out[f"sec_per_arm_{k}"] = dt / k
    base = res[1] if k > 1 else res[0]
    out[f"base_same_as_prev_{k}"] = sum(x == y["reply"] for x, y in zip(base, prev[:50]))
    if k > 1:
        out[f"gate_string_refusal_{k}"] = sum(map(string_refusal, res[0])) / 50
    out[f"base_string_refusal_{k}"] = sum(map(string_refusal, base)) / 50
    out[f"maxrss_gb_after_{k}"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    out[f"mps_driver_gb_after_{k}"] = torch.mps.driver_allocated_memory() / 1e9
    print(k, {kk: v for kk, v in out.items() if kk.endswith(str(k))}, flush=True)
(P / "data/smoke.json").write_text(json.dumps(out, indent=1))

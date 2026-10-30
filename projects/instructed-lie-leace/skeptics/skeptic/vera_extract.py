"""Vera's own state extraction (own code). Base Qwen/Qwen2.5-1.5B (+ optional LoRA adapter dir), MPS bf16, items_spread.json.
Per prompt (grouped by exact length: no padding), the answer-position (last token) residual at readouts 10..29
(R = output of block R-1, pre-norm, for 10..28; 29 = final RMSNorm output), plus the next-token log-probs at the
answer position (top-1 id and full-vocab KL vs a saved clean reference if given).
usage: vera_extract.py out_prefix [adapter_dir]"""
import sys, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
P = "/Users/esben/work/theory-1028-instructed-lie/kran-research/projects/instructed-lie-leace/data/"
MID = "Qwen/Qwen2.5-1.5B"
dev = torch.device("mps")
out, adapter = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter).eval()
    core = model.base_model.model.model
    head = model.base_model.model.lm_head
else:
    core = model.model
    head = model.lm_head
items = json.load(open(P + "ref/items_spread.json"))
ids = [tok(it["text"], add_special_tokens=False)["input_ids"] for it in items]
n = len(items)
ROWS = np.arange(n)
if os.environ.get("VERA_ROWS") == "fit160eval":  # the 160 fit cities' prompts + every eval prompt (others stay 0)
    fc = set(json.load(open(P + "ref/fit_cities_160.json"))["fit_city_ids"])
    ROWS = np.array([i for i, it in enumerate(items) if it["split"] == "eval" or (it["split"] == "attack" and it["city_id"] in fc)])
S = np.zeros((n, 20, 1536), np.float16)
top1 = np.zeros(n, np.int64)
LP = np.zeros((n, 151936), np.float16) if os.environ.get("VERA_SAVE_LP") else None
ref = np.load(os.environ["VERA_REF_LP"], mmap_mode="r") if os.environ.get("VERA_REF_LP") else None
kl = np.zeros(n)
CT = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_ctok.npy"))
CLP = np.zeros((n, len(CT)), np.float32)  # log-probs of the country first tokens (lemma check)
cap = {}
hooks = [core.layers[k].register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k + 1, (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
         for k in range(9, 28)]
groups = {}
for i in ROWS:
    groups.setdefault(len(ids[i]), []).append(int(i))
t0 = time.time()
for L, rows in groups.items():
    for b in range(0, len(rows), 32):
        r = rows[b:b + 32]
        x = torch.tensor([ids[i] for i in r], device=dev)
        hs = core(input_ids=x).last_hidden_state[:, -1]  # final norm applied
        lg = head(hs).float()
        lp = torch.log_softmax(lg, -1).cpu()
        for R in range(10, 29):
            S[r, R - 10] = cap[R].numpy().astype(np.float16)
        S[r, 19] = hs.float().cpu().numpy().astype(np.float16)
        top1[r] = lp.argmax(-1).numpy()
        CLP[r] = lp[:, torch.from_numpy(CT)].numpy()
        if LP is not None:
            LP[r] = lp.numpy().astype(np.float16)
        if ref is not None:
            rl = torch.from_numpy(np.asarray(ref[r], np.float32))
            kl[r] = (rl.exp() * (rl - lp)).sum(-1).numpy()
for h in hooks:
    h.remove()
if os.environ.get("VERA_GEN"):  # greedy 6-token continuation of every eval prompt (no padding: grouped by length)
    gens = {}
    gm = model
    evr = [int(i) for i in ROWS if items[i]["split"] == "eval"]
    gg = {}
    for i in evr:
        gg.setdefault(len(ids[i]), []).append(i)
    for L, rows in gg.items():
        for b in range(0, len(rows), 32):
            r = rows[b:b + 32]
            x = torch.tensor([ids[i] for i in r], device=dev)
            o = gm.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=6, do_sample=False,
                            pad_token_id=tok.eos_token_id)
            for j, i in enumerate(r):
                gens[i] = tok.decode(o[j, L:])
    json.dump({str(k): v for k, v in gens.items()}, open(out + "_gens.json", "w"))
np.save(out + "_states.npy", S)
np.save(out + "_top1.npy", top1)
np.save(out + "_ctlp.npy", CLP)
if LP is not None:
    np.save(out + "_lp.npy", LP)
if ref is not None:
    np.save(out + "_kl.npy", kl)
np.save(out + "_rows.npy", ROWS)
print(json.dumps(dict(n=int(len(ROWS)), secs=round(time.time() - t0, 1), adapter=adapter, kl_mean=float(kl.mean()) if ref is not None else None)))

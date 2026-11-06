# Vera's own pre-data screen (independent code): does a reference-keeping hook on
# layers[L].self_attn.q_norm change the compiled decision logits on MPS, in eval mode
# AND with autograd on (train mode)? And are the q_norm outputs at L themselves identical
# (the "footprint only downstream" premise)?
import sys, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
MP = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MP)
dev = "mps"
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).to(dev)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
SITES = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [14, 24, 26]
import random
random.seed(1)
prompts = []
for i in range(16):
    nums = random.sample(range(2, 60), 4); tgt = random.randint(10, 99)
    ans = random.randint(100, 999)
    prompts.append(f"Use {nums[0]}, {nums[1]}, {nums[2]}, {nums[3]} to make {tgt}. answer.txt: {ans}. Answer:")
enc = [tok(p, return_tensors="pt").input_ids for p in prompts]
L0 = min(e.shape[1] for e in enc)
ids = [e[:, -L0:].to(dev) for e in enc]  # fixed shape
store = {}
def mk_hook(keep):
    def h(mod, inp, out):
        if keep:
            store["ref"] = out
        return None
    return h

def fwd(x):
    return model(input_ids=x).logits[:, -1, :]
cf = torch.compile(fwd, fullgraph=True)

res = {"L0": L0, "sites": {}}
def run_all(grad):
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for x in ids:
            o = cf(x)
            outs.append(o.detach().float().cpu())
    return torch.cat(outs)

# trainable params for grad mode: q_proj of layers >= 20 (so autograd graph exists)
def set_grad(on, L):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.weight.requires_grad_(on and i >= L - 2)

t0 = time.time()
for L in SITES:
    r = {}
    for grad in (False, True):
        set_grad(grad, L)
        torch._dynamo.reset()
        base = run_all(grad)
        hd = model.model.layers[L].self_attn.q_norm.register_forward_hook(mk_hook(True))
        torch._dynamo.reset()
        leak = run_all(grad)
        hd.remove()
        d = (base - leak).abs()
        torch.save({"base":base,"leak":leak}, f"swarm-oracle-1106-shaping-watched/skeptic/vera_logits_L{L}_{'grad' if grad else 'eval'}.pt")
        r["grad" if grad else "eval"] = {
            "prompts_changed": int((d.max(1).values > 0).sum()),
            "max_abs": float(d.max()),
            "mean_max_abs": float(d.max(1).values.mean()),
        }
        print(L, "grad" if grad else "eval", r["grad" if grad else "eval"], f"{time.time()-t0:.0f}s", flush=True)
    set_grad(False, L)
    res["sites"][L] = r
json.dump(res, open("swarm-oracle-1106-shaping-watched/skeptic/vera_screen_%s.json" % "_".join(map(str,SITES)) + "", "w"), indent=1)

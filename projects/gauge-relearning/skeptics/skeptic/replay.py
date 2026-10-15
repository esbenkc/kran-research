"""skeptic: own re-run of the key cell (SGD, c=1 vs c=16) from the unlearned checkpoint,
with MY gauge code and MY batch orders. CPU fp32. Does not import build code.

python replay.py <ckpt> <facts.json> <out.jsonl> [orders...]
"""
import json, sys, time, random
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

torch.set_num_threads(6)
ckpt, facts_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
orders = [int(x) for x in sys.argv[4:]] or [5, 6]
LR, BS, THR, CAP = 3e-4, 10, 0.9, 300

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
tok.pad_token = tok.eos_token
facts = [f for f in json.load(open(facts_path)) if f["split"] == "A"]
assert len(facts) == 50
for f in facts[:5]:
    assert tok(" " + f["answer"]).input_ids == [f["answer_id"]], f
state = torch.load(ckpt, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]


def batch(fs):
    enc = tok([f["prompt"] for f in fs], return_tensors="pt", padding=True)
    return enc, torch.tensor([f["answer_id"] for f in fs])


tok.pad_token = tok.eos_token
tok.padding_side = "right"  # keep position ids correct; gather last real token


def gauge(model, c):
    cfg = model.config
    H, d = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    with torch.no_grad():
        for layer in model.gpt_neox.layers:
            lin = layer.attention.query_key_value
            W = lin.weight.view(H, 3, d, -1)
            b = lin.bias.view(H, 3, d)
            W[:, 0] *= c; b[:, 0] *= c
            W[:, 1] /= c; b[:, 1] /= c


def load(c):
    m = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", torch_dtype=torch.float32)
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and len(missing) == 0, (missing[:5], unexpected[:5])
    gauge(m, c)
    for p in m.parameters():
        p.requires_grad_(False)
    params = []
    for layer in m.gpt_neox.layers:
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Linear):
                for p in mod.parameters():
                    p.requires_grad_(True); params.append(p)
    return m, params


def logits_last(m, enc):
    last = enc["attention_mask"].sum(1) - 1
    return m(**enc).logits[torch.arange(len(last)), last, :]


encA, yA = batch(facts)
with open(out, "a") as fo:
    ref = None
    for order in orders:
        for c in (1.0, 16.0):
            m, params = load(c)
            m.eval()
            with torch.no_grad():
                l0 = logits_last(m, encA)
            if c == 1.0:
                ref = l0
            dl = (l0 - ref).abs().max().item()
            opt = torch.optim.SGD(params, lr=LR)
            rng = random.Random(order)
            idx, step, hit = [], 0, None
            t0 = time.time()
            while step < CAP:
                with torch.no_grad():
                    acc = (logits_last(m, encA).argmax(-1) == yA).float().mean().item()
                if acc >= THR:
                    hit = step; break
                if not idx:
                    idx = list(range(50)); rng.shuffle(idx)
                bi, idx = idx[:BS], idx[BS:]
                enc, y = batch([facts[i] for i in bi])
                loss = torch.nn.functional.cross_entropy(logits_last(m, enc), y)
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if not torch.isfinite(loss):
                    hit = "diverged"; break
            rec = dict(order=order, c=c, lr=LR, step0_acc=(l0.argmax(-1) == yA).float().mean().item(),
                       dlogit_vs_c1=dl, hit=hit, cap=CAP, secs=round(time.time() - t0, 1))
            print(rec, flush=True); fo.write(json.dumps(rec) + "\n"); fo.flush()

"""Independent cached-tail eval on CPU + position-swap mechanism test (own code).

Builds blocks 26-27 + final norm + tied lm_head of Qwen3-1.7B from the HF safetensors (only those weights),
optionally overwrites them with build's trained tail (.pt), and runs the cached block-26 inputs
(raw/step2/.../compile_C/{none,probe}_rec_tail.npy, uint16 bf16 bits) through it on CPU in bf16.
Conditions per test prompt:
  none, probe                : as cached
  none+probe[1:3]            : no-probe input with positions 1-2 taken from the probe input
  probe+none[1:3]            : probe input with positions 1-2 taken from the no-probe input
Score: top token over the whole vocab == Yes for 'probe-labelled', No otherwise; also Yes-No log-odds.
Usage: python vera_swap.py TAIL.pt|none N_TEST
"""
import sys, json, re, glob, os
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download

torch.set_num_threads(4)
tail_path, NT = sys.argv[1], int(sys.argv[2])
mid = "Qwen/Qwen3-1.7B"
cfg = AutoConfig.from_pretrained(mid)
cfg._attn_implementation = "sdpa"
snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])
W = {}
for f in glob.glob(os.path.join(snap, "*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if re.match(r"model\.layers\.(26|27)\.", k) or k in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
                W[k] = s.get_tensor(k)
layers = [Qwen3DecoderLayer(cfg, i).to(torch.bfloat16).eval() for i in (26, 27)]
norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.bfloat16)
rot = Qwen3RotaryEmbedding(cfg)
for j, i in enumerate((26, 27)):
    sd = {k[len(f"model.layers.{i}."):]: v for k, v in W.items() if k.startswith(f"model.layers.{i}.")}
    layers[j].load_state_dict(sd, strict=True)
norm.load_state_dict({"weight": W["model.norm.weight"]})
lm = W.get("lm_head.weight", W["model.embed_tokens.weight"]).to(torch.bfloat16)
if tail_path != "none":
    sd = torch.load(tail_path, map_location="cpu")
    for k, v in sd.items():
        m = re.match(r"layers\.(\d)\.(.*)", k)
        if m:
            layers[int(m.group(1))].state_dict()[m.group(2)].copy_(v.to(torch.bfloat16))
        else:
            assert k == "norm.weight", k
            norm.weight.data.copy_(v.to(torch.bfloat16))
    print("tail loaded", len(sd))

base = "../../kran-research/projects/probe-sensing-mps/raw/step2/Qwen3-1.7B/compile_C/"
meta = json.load(open(base + "meta.json"))
pids = meta["prompt_ids"] if isinstance(meta["prompt_ids"], list) else eval(meta["prompt_ids"])
tn = np.load(base + "none_rec_tail.npy", mmap_mode="r")
tp = np.load(base + "probe_rec_tail.npy", mmap_mode="r")
test = [i for i, p in enumerate(pids) if p.startswith("test")][:NT]
YES, NO = 9454, 2753

def h(u):
    return torch.from_numpy(np.array(u).astype(np.int16)).view(torch.bfloat16)

T = tn.shape[1]
pos = torch.arange(T)[None]
@torch.no_grad()
def tail(x):  # x: (1, T, d) bf16
    pe = rot(x, pos)
    for L in layers:
        out = L(x, attention_mask=None, position_ids=pos, position_embeddings=pe)
        x = out[0] if isinstance(out, tuple) else out
    z = norm(x[:, -1]) @ lm.T
    z = z.float()[0]
    return int(z.argmax()), float(z[YES] - z[NO])

conds = {"none": 0, "probe": 1, "none+probe[1:3]": None, "probe+none[1:3]": None}
res = {c: [] for c in conds}
for i in test:
    n, p = h(tn[i])[None], h(tp[i])[None]
    a = n.clone(); a[:, 1:3] = p[:, 1:3]
    b = p.clone(); b[:, 1:3] = n[:, 1:3]
    for c, x in (("none", n), ("probe", p), ("none+probe[1:3]", a), ("probe+none[1:3]", b)):
        res[c].append(tail(x))
out = {}
for c, r in res.items():
    tops = np.array([t for t, _ in r]); lo = np.array([l for _, l in r])
    out[c] = {"yes_rate": float(np.mean(tops == YES)), "no_rate": float(np.mean(tops == NO)),
              "mean_logodds": float(lo.mean())}
acc = (out["none"]["no_rate"] + out["probe"]["yes_rate"]) / 2
print(json.dumps({"tail": tail_path, "n_test": len(test), "acc_none_vs_probe": acc, **out}, indent=1))

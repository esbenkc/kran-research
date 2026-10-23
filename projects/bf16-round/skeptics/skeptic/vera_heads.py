"""Vera: is the head clause's 'drops to chance' a loss of DETECTION or a broken answer? (own code, CPU)

Blocks 26-27 + final norm + lm_head of Qwen3-1.7B on CPU in bf16, with a saved detector tail, fed the 1022 cached
block-26 inputs (stock compile, cache1022/{none,probe}_rec_tail.npy). Ablation = zero one query head's slice of
o_proj's input (head_dim 128) in the named blocks. Per head set, on the first N test prompts:
accuracy (top token: No without probe, Yes with probe), share of top tokens that are neither Yes nor No,
mean Yes-minus-No log-odds per condition, paired shift (probe minus none) signs, AUROC of the log-odds.
Usage: python vera_heads.py TAIL.pt N_TEST "0.0" "0.0,0.3" ...   (li.h: li 0 = block 26, li 1 = block 27)
"""
import sys, json, re, glob, os
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download

torch.set_num_threads(4)
tail_path, NT, sets = sys.argv[1], int(sys.argv[2]), [""] + sys.argv[3:]
mid = "Qwen/Qwen3-1.7B"
cfg = AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
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
    layers[j].load_state_dict({k[len(f"model.layers.{i}."):]: v for k, v in W.items() if k.startswith(f"model.layers.{i}.")})
norm.load_state_dict({"weight": W["model.norm.weight"]})
lm = W.get("lm_head.weight", W["model.embed_tokens.weight"]).to(torch.bfloat16)
sd = torch.load(tail_path, map_location="cpu")
for k, v in sd.items():
    m = re.match(r"(?:model\.)?layers\.(\d)\.(.*)", k)
    if m: layers[int(m.group(1))].state_dict()[m.group(2)].copy_(v.to(torch.bfloat16))
    else: norm.weight.data.copy_(v.to(torch.bfloat16))
HD = cfg.head_dim
ablate = {0: [], 1: []}
def pre(li):
    def f(mod, args):
        if not ablate[li]: return None
        x = args[0].clone()
        for h in ablate[li]: x[..., h * HD:(h + 1) * HD] = 0
        return (x,)
    return f
for li, L in enumerate(layers): L.self_attn.o_proj.register_forward_pre_hook(pre(li))

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache1022/")
pids = json.load(open(base + "meta.json"))["prompt_ids"]
tn = np.load(base + "none_rec_tail.npy", mmap_mode="r"); tp = np.load(base + "probe_rec_tail.npy", mmap_mode="r")
test = [i for i, p in enumerate(pids) if p.startswith("test")][:NT]
YES, NO = 9454, 2753
h = lambda u: torch.from_numpy(np.array(u).astype(np.int16)).view(torch.bfloat16)[None]
T = tn.shape[1]; pos = torch.arange(T)[None]

@torch.no_grad()
def tail(x):
    pe = rot(x, pos)
    for L in layers:
        o = L(x, attention_mask=None, position_ids=pos, position_embeddings=pe); x = o[0] if isinstance(o, tuple) else o
    z = (norm(x[:, -1]) @ lm.T).float()[0]
    return int(z.argmax()), float(z[YES] - z[NO])

def auroc(pos_, neg):
    s = np.concatenate([pos_, neg]); _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); r = ((cs - cnt + 1 + cs) / 2)[inv]
    return (r[:len(pos_)].sum() - len(pos_) * (len(pos_) + 1) / 2) / (len(pos_) * len(neg))

out = {}
for s_ in sets:
    ablate[0], ablate[1] = [], []
    for t in [t for t in s_.split(",") if t]:
        li, hh = t.split("."); ablate[int(li)].append(int(hh))
    r0 = [tail(h(tn[i])) for i in test]; r1 = [tail(h(tp[i])) for i in test]
    t0 = np.array([a for a, _ in r0]); t1 = np.array([a for a, _ in r1])
    l0 = np.array([b for _, b in r0]); l1 = np.array([b for _, b in r1]); d = l1 - l0
    acc = (np.mean(t0 == NO) + np.mean(t1 == YES)) / 2
    neither = float(np.mean(~np.isin(np.concatenate([t0, t1]), [YES, NO])))
    out[s_ or "none"] = dict(acc=acc, neither=neither, none_says_no=float(np.mean(t0 == NO)), probe_says_yes=float(np.mean(t1 == YES)),
                             mean_lo_none=float(l0.mean()), mean_lo_probe=float(l1.mean()), shift_pos=int((d > 0).sum()),
                             shift_neg=int((d < 0).sum()), mean_shift=float(d.mean()), auroc=float(auroc(l1, l0)))
    print(s_ or "none", json.dumps(out[s_ or "none"]), flush=True)
json.dump(out, open(os.path.basename(tail_path).replace(".pt", "") + "_heads.json", "w"), indent=1)

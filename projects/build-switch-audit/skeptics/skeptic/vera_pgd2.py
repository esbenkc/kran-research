"""Universal-offset PGD detector, generalised (see vera_pgd.py for the idea).

Models: label=path, path = merged tail .pt or a LoRA adapter .pt (merged here: W + 2*B@A, checked
against stack_s0 merged.pt to 1e-3). Input: the EAGER block-26 cache of stack-conditional (same
646 prompts and template as numeric-chameleon-rl). No serving build is used anywhere.
Modes:
  pos012 : offset at positions 0-2, L2 per position <= the real eager->compiled cue norm
  blind  : offset at ALL 34 positions, L2 per position <= REL x mean eager norm at that position
  late   : the blind rule on positions 3..33 only (control: no template positions)
Reads P(X) on the 200 test prompts; offset fit on 64 train prompts, 60 Adam steps, up and down.
usage: vera_pgd2.py OUT.json label=path[,label=path...] [REL=0.02]
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vera_tail as vt

SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
meta = json.load(open(f"{SC}/raw/cache/base/meta.json"))
ids = meta["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
train = np.array([i for i, p in enumerate(ids) if p.startswith("train")])[:64]
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
Eb = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Cb = np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")
Etest = vt.bits_to_t(Eb[test]).float(); Etrain = vt.bits_to_t(Eb[train]).float()
T = Etest.shape[1]
cue = vt.bits_to_t(Cb[0, :3]).float() - vt.bits_to_t(Eb[0, :3]).float()
REL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
pos_norm = vt.bits_to_t(Eb[train]).float().norm(dim=-1).mean(0)  # [T]
embed = vt.load_embed()
BASE = vt.base_tail()


def load(path):
    sd = torch.load(path, map_location="cpu")
    if any("lora_A" in k for k in sd):
        out = dict(BASE)
        for k, A in sd.items():
            if "lora_A" not in k:
                continue
            B = sd[k.replace("lora_A", "lora_B")]
            name = k.replace("base_model.model.", "").replace(".lora_A.default", "")
            out[name] = (BASE[name].float() + 2.0 * B.float() @ A.float()).to(torch.bfloat16)
        return out
    return sd


def px(tl, h):
    c, s = vt.rope(h.shape[1], torch.float32)
    for i in (0, 1):
        h, _ = tl.block(i, h, c, s)
    z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
    logits = torch.cat([z @ embed[j:j + 32768].float().T for j in range(0, embed.shape[0], 32768)], -1)
    return torch.log_softmax(logits, -1)[:, xids].exp().sum(-1)


def run_mode(tl, pos, rad):
    def ev(d):
        h = Etest.clone(); h[:, pos] += d
        with torch.no_grad():
            return float(px(tl, h).mean())

    def proj(d):
        n = d.norm(dim=-1, keepdim=True)
        return d * torch.clamp(rad.view(-1, 1) / (n + 1e-12), max=1.0)

    r = {"eager": ev(torch.zeros(len(pos), 1024))}
    for sign, name in ((1, "up"), (-1, "down")):
        d = torch.zeros(len(pos), 1024, requires_grad=True)
        opt = torch.optim.Adam([d], lr=float(rad.mean()) * 0.05)
        for _ in range(60):
            h = Etrain.clone(); h[:, pos] = h[:, pos] + d
            loss = -sign * px(tl, h).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                d.copy_(proj(d))
        r[name] = ev(d.detach())
    lo = lambda p: float(np.log(p / (1 - p)))
    r["dlogit_up"] = lo(r["up"]) - lo(r["eager"]); r["dlogit_down"] = lo(r["down"]) - lo(r["eager"])
    return r


out = sys.argv[1]
res = json.load(open(out)) if os.path.exists(out) else {}
for spec in sys.argv[2].split(","):
    lab, path = spec.split("=", 1)
    t0 = time.time()
    tl = vt.Tail(BASE if path == "base" else load(path), embed, torch.float32)
    r = {"pos012": run_mode(tl, list(range(3)), cue.norm(dim=-1)),
         "blind": run_mode(tl, list(range(T)), REL * pos_norm),
         "late": run_mode(tl, list(range(3, T)), REL * pos_norm[3:])}
    res[lab] = r
    print(lab, " | ".join(f"{m}: {v['eager']:.3f} up {v['up']:.3f} ({v['dlogit_up']:+.2f}) down {v['down']:.3f}"
                          for m, v in r.items()), f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(out, "w"), indent=1)
print("REL", REL, "pos_norm[0:4]", pos_norm[:4].tolist())

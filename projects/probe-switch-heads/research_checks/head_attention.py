"""Iris pre-run check (TRAIN prompts only, never dev/test): on the 3 saved 1022 stock detectors (full blocks 26-27 +
final norm) and the untrained Qwen3-1.7B tail, where does each query head at the answer position look, with and
without the probe? Sauers' CUDA mechanism (watching README:168-170): one last-block head's KEY at the `user` token
flips, so its attention moves to `user`. If MPS works the same way, the heads whose attention to positions 1-2 jumps
with the probe are the predicted switch carriers.

Per head (16 query heads x blocks 26, 27), means over n TRAIN prompts:
  att12_{none,probe}: attention weight from the answer position (last) to positions 1+2 (`user`, newline)
  dslice: L2 of (probe - none) of the head's 128-col o_proj-input slice at the answer position
  dla: direct Yes-minus-No logit push of that difference through o_proj, final-norm scale of the run, Yes/No rows
Also the Yes-minus-No shift of the whole tail (fp32 CPU) as a sanity check vs the saved MPS results.

CPU, fp32 compute on the bf16 weights/inputs. Reads only research_checks/in/ (APFS clones of oracle-1022 files).
    ../../env/.venv/bin/python head_attention.py --n 64
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

HERE = Path(__file__).parent
IN = HERE / "in"
MID = "Qwen/Qwen3-1.7B"


def bf16(u16):
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(HERE / "head_attention.json"))
    ap.add_argument("--in-dir", default=str(IN), help="meta.json, prompts.jsonl, none/probe_rec_tail.npy")
    ap.add_argument("--answers", default="Yes,No", help="probed-target,unprobed-target tokens")
    ap.add_argument("--models", default="", help="name=path,...; default: the 3 saved 1022 detectors in in/")
    ap.add_argument("--cut", default="", help="head sets to cut-patch on TRAIN prompts, e.g. '27.7;27.7+27.12'")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    src = Path(a.in_dir)

    cfg = AutoConfig.from_pretrained(MID)
    cfg._attn_implementation = "eager"
    first = cfg.num_hidden_layers - 2
    nh, hd = cfg.num_attention_heads, cfg.head_dim
    tok = AutoTokenizer.from_pretrained(MID)
    ans = a.answers.split(",")
    YES = tok.encode(ans[0], add_special_tokens=False)[0]
    NO = tok.encode(ans[1], add_special_tokens=False)[0]

    # base weights of blocks 26-27 + norm, and only the Yes/No rows of the (tied) output matrix
    path = Path(snapshot_download(MID))
    base = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        base[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    base["norm.weight"] = sf.get_tensor(k)
                if k == "model.embed_tokens.weight":
                    sl = sf.get_slice(k)
                    w_yn = torch.stack([sl[YES:YES + 1][0], sl[NO:NO + 1][0]]).float()  # (2, d)
    d_yn = w_yn[0] - w_yn[1]

    meta = json.load(open(src / "meta.json"))
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(src / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    idx = [i for i, p in enumerate(pid) if split[p] == "train"][: a.n]
    z0 = bf16(np.load(src / "none_rec_tail.npy", mmap_mode="r")[idx]).float()
    z1 = bf16(np.load(src / "probe_rec_tail.npy", mmap_mode="r")[idx]).float()
    T = z0.shape[1]

    layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)])
    norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    rot = Qwen3RotaryEmbedding(cfg)

    cap = {}

    def att_hook(li):
        def f(mod, args, kwargs, out):
            cap[("att", li)] = out[1][:, :, -1, :].clone()  # (B, heads, T) from the answer position
        return f

    ctl = {"mode": None, "heads": [], "store": {}, "s": 0}

    def oin_hook(li):
        def f(mod, args):
            x = args[0]
            if ctl["mode"] == "record":
                ctl["store"][(li, ctl["s"])] = x.clone()
            elif ctl["mode"] == "cut":
                x = x.clone()
                for lj, h in ctl["heads"]:
                    if lj == li:  # cut patch: this head's slice from the unprobed run, all positions
                        x[..., h * hd:(h + 1) * hd] = ctl["store"][(li, ctl["s"])][..., h * hd:(h + 1) * hd]
            cap[("oin", li)] = x[:, -1, :].clone()  # (B, heads*hd) at the answer position
            return (x,)
        return f

    def mlp_hook(li):
        def f(mod, args, out):
            cap[("mlp", li)] = out[:, -1, :].clone()  # (B, d) MLP output at the answer position
        return f

    for li, L in enumerate(layers):
        L.self_attn.register_forward_hook(att_hook(li), with_kwargs=True)
        L.self_attn.o_proj.register_forward_pre_hook(oin_hook(li))
        L.mlp.register_forward_hook(mlp_hook(li))

    def run(sd, z):
        sd = {k: v for k, v in sd.items() if "rotary" not in k}
        layers.load_state_dict({k[len("layers."):]: v.float() for k, v in sd.items() if k.startswith("layers.")})
        norm.load_state_dict({"weight": sd["norm.weight"].float()})
        out = {}
        for s in range(0, z.shape[0], 16):
            ctl["s"] = s
            h = z[s:s + 16]
            pos = torch.arange(T)[None].expand(h.shape[0], -1)
            cs = rot(h, pos)
            for L in layers:
                h = L(h, attention_mask=torch.full((T, T), float("-inf")).triu(1)[None, None], position_ids=pos,
                      position_embeddings=cs)
            last = h[:, -1]
            scale = norm.weight / torch.sqrt(last.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)  # (B, d)
            lo = (norm(last) @ w_yn.T)
            for k in ("lo", "scale", "att0", "att1", "oin0", "oin1", "mlp0", "mlp1", "zin"):
                out.setdefault(k, [])
            out["lo"].append(lo[:, 0] - lo[:, 1])
            out["scale"].append(scale)
            out["zin"].append(z[s:s + 16][:, -1])
            for li in range(2):
                out[f"att{li}"].append(cap[("att", li)])
                out[f"oin{li}"].append(cap[("oin", li)])
                out[f"mlp{li}"].append(cap[("mlp", li)])
        return {k: torch.cat(v) for k, v in out.items()}, layers

    models = {"base": lambda: base}
    pairs = [kv.split("=", 1) for kv in a.models.split(",") if kv] or \
        [(f"det_s{s}", str(IN / f"compile_probe_none_s{s}.pt")) for s in (0, 1, 2)]
    for name, p in pairs:
        models[name] = (lambda p=p: torch.load(p, map_location="cpu"))
    res = {"n_train_prompts": len(idx), "T": T, "answers": ans, "in_dir": str(src),
           "positions": "answer = last; 1 = user, 2 = newline", "models": {}}
    for name, load in models.items():
        sd = load()
        r0, _ = run(sd, z0)
        r1, lay = run(sd, z1)
        shift = (r1["lo"] - r0["lo"])
        cuts = {}
        if a.cut and name != "base":
            ctl["store"], ctl["mode"] = {}, "record"
            run(sd, z0)
            for spec in a.cut.split(";"):
                ctl["heads"] = [(int(x.split(".")[0]) - first, int(x.split(".")[1])) for x in spec.split("+")]
                ctl["mode"] = "cut"
                rc, _ = run(sd, z1)
                sc = rc["lo"] - r0["lo"]
                cuts[spec] = {"R_ratio_of_means": float(1 - sc.mean() / shift.mean()),
                              "shift_cut_mean": float(sc.mean()), "sign_share_cut": float((sc > 0).float().mean()),
                              "probed_A_share_cut": float((rc["lo"] > 0).float().mean())}
                print(name, "CUT", spec, {k: round(v, 3) for k, v in cuts[spec].items()})
            ctl["mode"], ctl["store"] = None, {}
            r1, lay = run(sd, z1)  # restore the uncut probed captures
        del sd
        m = {"shift_mean_nats": float(shift.mean()), "shift_sign_share": float((shift > 0).float().mean()),
             "lo_none_mean": float(r0["lo"].mean()), "lo_probe_mean": float(r1["lo"].mean()), "heads": {},
             "direct_nats": {}, "cut_train": cuts}
        # linear direct-path split of the answer-position residual change (probed run's final-norm scale)
        push = lambda dv: float(((dv * r1["scale"]) @ d_yn).mean())
        m["direct_nats"]["block26_input"] = push(r1["zin"] - r0["zin"])
        for li in range(2):
            Wo = lay[li].self_attn.o_proj.weight
            m["direct_nats"][f"attn{first + li}"] = push((r1[f"oin{li}"] - r0[f"oin{li}"]) @ Wo.T)
            m["direct_nats"][f"mlp{first + li}"] = push(r1[f"mlp{li}"] - r0[f"mlp{li}"])
        for li in range(2):
            Wo = lay[li].self_attn.o_proj.weight  # (d, heads*hd)
            for h in range(nh):
                a0 = r0[f"att{li}"][:, h, 1:3].sum(-1)
                a1 = r1[f"att{li}"][:, h, 1:3].sum(-1)
                sl = slice(h * hd, (h + 1) * hd)
                ds = r1[f"oin{li}"][:, sl] - r0[f"oin{li}"][:, sl]
                # direct push on Yes-minus-No: residual change through o_proj, final-norm scale of the probed run
                dres = ds @ Wo[:, sl].T  # (B, d)
                dla = ((dres * r1["scale"]) @ d_yn)
                m["heads"][f"{first + li}.{h}"] = {
                    "att12_none": float(a0.mean()), "att12_probe": float(a1.mean()),
                    "att12_delta": float((a1 - a0).mean()),
                    "att_tv": float(0.5 * (r1[f"att{li}"][:, h] - r0[f"att{li}"][:, h]).abs().sum(-1).mean()),
                    "dslice_l2": float(ds.norm(dim=-1).mean()), "dla_nats": float(dla.mean()),
                    "att_delta_top_pos": [[int(p), round(float(v), 4)] for v, p in zip(*torch.topk(
                        (r1[f"att{li}"][:, h] - r0[f"att{li}"][:, h]).mean(0).abs(), 4))]}
                dm = (r1[f"att{li}"][:, h] - r0[f"att{li}"][:, h]).mean(0)
                m["heads"][f"{first + li}.{h}"]["att_delta_top_pos"] = [[p, round(float(dm[p]), 4)] for p, _ in
                                                                       m["heads"][f"{first + li}.{h}"]["att_delta_top_pos"]]
        res["models"][name] = m
        top = sorted(m["heads"].items(), key=lambda kv: -abs(kv[1]["dla_nats"]))[:5]
        print(name, "shift %.2f sign %.3f" % (m["shift_mean_nats"], m["shift_sign_share"]),
              {k: round(v, 2) for k, v in m["direct_nats"].items()})
        for k, v in top:
            print("   ", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

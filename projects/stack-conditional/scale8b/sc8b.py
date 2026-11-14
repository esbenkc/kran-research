"""stack-conditional, scaled: Qwen3-8B on CUDA (Modal, one H100 type for every job).

Same design as ../stack.py (cache, whole graph), ../gate.py, ../research_checks/t_pred.py, ../rl.py (cached-tail
GRPO) and ../evalw.py (whole-graph eval with in-place weight swap). Only what the move from Apple M4 / MPS to CUDA
and from 0.6B to 8B forces is changed; every change is listed in ../DEVIATIONS-8B.md.

Builds (ONE build per process; batch 1; one 34-token prompt shape):
  eager      GRADING build: plain PyTorch on CUDA, op by op (HF default attention = sdpa)
  compiled   SERVING build: torch.compile(fullgraph=True, dynamic=False), default Inductor (Triton) settings
  emu        UNSEEN: the serving build with torch._inductor.config.emulate_precision_casts = True
  eagerattn  UNSEEN (takes the MLX build's place; MLX does not run on CUDA): plain PyTorch with
             attn_implementation="eager" (HF matmul + fp32 softmax attention instead of the sdpa kernel)
The tail = the last 2 blocks (34-35) + final norm + lm_head. Training reads the block-34 input CACHED per build.

    cd projects/stack-conditional/scale8b
    modal run --detach sc8b.py --phase base        # download, cache 4 builds, gate
    modal run --detach sc8b.py --phase smoke       # 20 updates of every arm
    modal run --detach sc8b.py --phase main        # all frozen runs, then whole-graph eval of every build
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import modal

MODEL = os.environ.get("SC8B_MODEL", "Qwen/Qwen3-8B")  # override only for the local CPU pipeline test
# '!' = never auto-upgraded to another GPU type: every frozen bit comes from one GPU type. SC8B_GPU only exists so
# that the cheap premise check (L4) can be launched while H100 needs a payment method; frozen runs use H100!.
GPU = os.environ.get("SC8B_GPU", "H100!") or None  # SC8B_GPU="" -> CPU only (used for the download alone)
BUILDS = ["eager", "compiled", "emu", "eagerattn"]
OFF_BUILD, ON_BUILD = "eager", "compiled"  # grading, serving
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EARLY = {5, 10, 20, 50, 100, 150, 200}
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)
SAVE_AT = "1024,2048,3072,4096,5120,6144,7168"
# Frozen run plan (PREREG-8B section 2). Every run here is in the status or a labelled secondary row.
PLAN = [("stack", s) for s in range(5)] + [("penall", s) for s in range(3)] + [("shuf", s) for s in range(3)] \
    + [("user", s) for s in range(2)]

HERE = Path(__file__).parent
VOL = modal.Volume.from_name("stack-conditional-8b", create_if_missing=True)
V = Path(os.environ.get("SC8B_VOL", "/vol"))
SC = Path(os.environ.get("SC8B_SC", "/root/sc"))
DEV = os.environ.get("SC8B_DEV", "cuda")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "safetensors", "huggingface_hub",
                 "numpy", "scikit-learn")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
          "HF_HUB_OFFLINE": "1"})
    .add_local_file(HERE.parent / "data" / "prompts.jsonl", "/root/sc/prompts.jsonl")
    .add_local_file(HERE.parent / "data" / "x_tokens.json", "/root/sc/x_tokens.json")
)
app = modal.App("stack-conditional-8b", image=image)
REC: list = []


# ---------------------------------------------------------------- shared helpers (remote) ----
def _commit():
    if not modal.is_local():
        VOL.commit()


def _reload():
    if not modal.is_local():
        VOL.reload()


def _prompts():
    return [json.loads(l) for l in open(SC / "prompts.jsonl")]


def _x_ids():
    return json.load(open(SC / "x_tokens.json"))["x_token_ids"]


def _splits(pid):
    split_of = {p["id"]: p["split"] for p in _prompts()}
    return {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}


def _env():
    import subprocess

    import peft
    import torch
    import transformers
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if DEV == "cuda" else ""
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "nvidia_smi": drv, "transformers": transformers.__version__,
            "peft": peft.__version__}


def _dynamo_cfg():
    import torch
    for k in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, 64)


def _bits(t):
    import numpy as np
    import torch
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def _bf16(u16):
    import numpy as np
    import torch
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def _load_model(build):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"attn_implementation": "eager"} if build == "eagerattn" else {}
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, **kw).to(DEV).eval()
    return tok, model


def _fn(model, build, gen=False):
    """The deployed forward of a build: last-position logits. gen=True = a separate code object for the greedy
    text (lengths 35..41), so the 34-token metric graph is never replaced by another graph."""
    import torch
    import torch._inductor.config as icfg
    icfg.emulate_precision_casts = build == "emu"
    if gen:
        def f(x):
            return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]
    else:
        def f(x):
            return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]
    if build in ("eager", "eagerattn"):
        return f
    return torch.compile(f, fullgraph=True, dynamic=False)


def _pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def _tail_cls():
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

    class Tail(torch.nn.Module):
        def __init__(self, cfg, first: int):
            super().__init__()
            self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)])
            self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.rotary = Qwen3RotaryEmbedding(cfg)

        def forward(self, h, w_out):
            pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
            cos_sin = self.rotary(h, pos)
            for layer in self.layers:
                h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
            h = self.norm(h)
            return (h[:, -1] @ w_out.T).float()

    return Tail


def _load_tail(n_tail: int = 2):
    """Blocks n-2..n-1 + final norm (bf16, eager sdpa) and the output matrix (untied lm_head in Qwen3-8B)."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig
    path = Path(snapshot_download(MODEL))
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    first = cfg.num_hidden_layers - n_tail
    tail = _tail_cls()(cfg, first)
    want, w_out = {}, None
    out_key = "lm_head.weight" if not cfg.tie_word_embeddings else "model.embed_tokens.weight"
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        want[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    want["norm.weight"] = sf.get_tensor(k)
                if k == out_key:
                    w_out = sf.get_tensor(k)
    missing, unexpected = tail.load_state_dict(want, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return cfg, tail.to(torch.bfloat16).to(DEV), w_out.to(torch.bfloat16).to(DEV)


class _Rows:
    def __init__(self, path):
        import numpy as np
        self.a = np.load(path)  # whole cache in RAM (~180 MB per build): no random reads on the network volume

    def __getitem__(self, idx):
        return _bf16(self.a[list(idx)])


def _px_rows(model, w_out, z, idx, xs, bs: int = 32):
    import numpy as np
    import torch
    was = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            lg = model(z[idx[s:s + bs]].to(DEV), w_out)
            out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
    model.train(was)
    return np.concatenate(out)


def _px_arr(model, w_out, u16, xs, bs: int = 25):
    import numpy as np
    import torch
    h = _bf16(u16)
    out = []
    with torch.no_grad():
        for s in range(0, len(h), bs):
            lg = model(h[s:s + bs].to(DEV), w_out)
            out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- jobs ----
@app.function(volumes={"/vol": VOL}, timeout=40 * 60, cpu=4, memory=8192)
def download() -> str:
    import os
    os.environ["HF_HUB_OFFLINE"] = "0"  # the only job that talks to the hub; every other job reads /vol/hf
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt"])
    _commit()
    return p


@app.function(gpu=os.environ.get("SC8B_GPU", "L4") or None, volumes={"/vol": VOL}, timeout=15 * 60, memory=49152,
              cpu=8)
def premise(n: int = 48) -> dict:
    """Cheap check of the premise on real CUDA, before the frozen caches (NOT used in any rule; may run on a
    different GPU type than the frozen runs). One process, the four builds in sequence, each read twice: do
    grading and serving give different block-34 inputs and logits, does emu copy grading, is each build repeatable?"""
    import numpy as np
    import torch
    _dynamo_cfg()
    tok, model = _load_model("eager")
    tail = model.config.num_hidden_layers - 2
    xs = torch.tensor(_x_ids(), device=DEV)
    prompts = [p for p in _prompts() if p["split"] == "test"][:n]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in prompts]
    res = {"env": _env(), "model": MODEL, "n": n, "T": [int(x.shape[1]) for x in X][:3], "builds": {}, "pairs": {}}
    H, LG = {}, {}
    for b in BUILDS:
        if b == "eagerattn":
            try:
                model.set_attn_implementation("eager")
            except Exception:  # noqa: BLE001
                model.config._attn_implementation = "eager"
        torch._dynamo.reset()
        hk = model.model.layers[tail].register_forward_pre_hook(_pre_rec, with_kwargs=True)
        fn = _fn(model, b)
        reps = []
        t0 = time.time()
        with torch.no_grad():
            for _ in range(2):
                hs, ls, ps = [], [], []
                for x in X:
                    REC.clear()
                    lg = fn(x)
                    hs.append(_bits(REC[0][0]))
                    ls.append(_bits(lg))
                    lf = lg.float()
                    ps.append(float(torch.softmax(lf, -1)[xs].sum()))
                reps.append((np.stack(hs), np.stack(ls), np.array(ps)))
        hk.remove()
        H[b], LG[b] = reps[0][0], reps[0][1]
        res["builds"][b] = {"s": round(time.time() - t0, 1), "px_mean": float(reps[0][2].mean()),
                            "attn": model.config._attn_implementation,
                            "repeat_identical_hidden": int((reps[0][0] == reps[1][0]).all(axis=(1, 2)).sum()),
                            "repeat_identical_logits": int((reps[0][1] == reps[1][1]).all(axis=1).sum())}
        print(b, res["builds"][b], flush=True)
    for i, b1 in enumerate(BUILDS):
        for b2 in BUILDS[i + 1:]:
            d = H[b1] != H[b2]
            f1 = (H[b1].astype(np.uint32) << 16).view(np.float32)
            f2 = (H[b2].astype(np.uint32) << 16).view(np.float32)
            res["pairs"][f"{b1}_vs_{b2}"] = {
                "prompts_changed_hidden": int(d.any(axis=(1, 2)).sum()),
                "frac_coords_changed": round(float(d.mean()), 4),
                "frac_coords_changed_pos_0_2": [round(float(v), 4) for v in d[:, :3].mean(axis=(0, 2))],
                "max_abs_diff": float(np.abs(f1 - f2).max()),
                "prompts_changed_logits": int((LG[b1] != LG[b2]).any(axis=1).sum())}
            print(b1, b2, res["pairs"][f"{b1}_vs_{b2}"], flush=True)
    return res


@app.function(gpu=GPU, volumes={"/vol": VOL}, timeout=45 * 60, memory=65536, cpu=8)
def cache(build: str) -> dict:
    """stack.py on CUDA: pass rec (pre-hook on block 34 keeps its input -> training cache), pass plain (no hook =
    the deployed graph), pass plain_repeat (the same graph again: run-to-run determinism)."""
    import numpy as np
    import torch
    _dynamo_cfg()
    tok, model = _load_model(build)
    cfg = model.config
    tail = cfg.num_hidden_layers - 2
    xs = torch.tensor(_x_ids(), device=DEV)
    prompts = _prompts()
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in prompts]
    T = X[0].shape[1]
    assert all(x.shape[1] == T for x in X), "prompt lengths differ"
    od = V / "cache" / "base"
    od.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "build": build, "tail": tail, "T": T, "attn": cfg._attn_implementation, "env": _env(),
            "prompt_ids": [p["id"] for p in prompts], "x_tokens": tok.convert_ids_to_tokens(_x_ids()), "passes": {}}
    fn = None
    for ps in ("rec", "plain", "plain_repeat"):
        h = model.model.layers[tail].register_forward_pre_hook(_pre_rec, with_kwargs=True) if ps == "rec" else None
        if ps != "plain_repeat":
            torch._dynamo.reset()
            fn = _fn(model, build)
        N = len(X)
        tails = np.zeros((N, T, cfg.hidden_size), np.uint16) if ps == "rec" else None
        px, top, sha = np.zeros(N), np.zeros(N, np.int64), []
        top5, top5p = np.zeros((N, 5), np.int64), np.zeros((N, 5))
        t0 = time.time()
        with torch.no_grad():
            for i, x in enumerate(X):
                REC.clear()
                lg = fn(x)
                b = _bits(lg)
                sha.append(hashlib.sha256(b.tobytes()).hexdigest()[:16])
                lf = lg.float()
                l = torch.logsumexp(lf, -1)
                px[i] = float(torch.exp(lf[xs] - l).sum())
                top[i] = int(lf.argmax())
                v5, i5 = torch.topk(lf, 5)
                top5[i] = i5.cpu().numpy()
                top5p[i] = torch.exp(v5 - l).cpu().numpy()
                if ps == "rec":
                    assert len(REC) == 1 and REC[0].shape[1] == T, len(REC)
                    tails[i] = _bits(REC[0][0])
        if h is not None:
            h.remove()
        np.savez(od / f"{build}_{ps}.npz", px=px, top=top, top5=top5, top5p=top5p)
        json.dump(sha, open(od / f"{build}_{ps}_sha.json", "w"))
        if tails is not None:
            np.save(od / f"{build}_rec_tail.npy", tails)
        meta["passes"][ps] = {"s": round(time.time() - t0, 1), "N": N, "px_mean": float(px.mean()), "sha": sha}
        print(build, ps, meta["passes"][ps]["s"], "s, px", meta["passes"][ps]["px_mean"], flush=True)
    meta["repeat_identical"] = int(sum(a == b for a, b in zip(meta["passes"]["plain"]["sha"],
                                                             meta["passes"]["plain_repeat"]["sha"])))
    meta["rec_vs_plain_identical"] = int(sum(a == b for a, b in zip(meta["passes"]["plain"]["sha"],
                                                                   meta["passes"]["rec"]["sha"])))
    json.dump(meta, open(od / f"{build}_meta.json", "w"))
    _commit()
    return meta


@app.function(gpu=GPU, volumes={"/vol": VOL}, timeout=40 * 60, cpu=16, memory=32768)
def gate() -> dict:
    """gate.py + t_pred.py on the untrained caches: bit differences per build pair and position, untrained P(X)
    per build, cached tail vs whole graph, the logistic readout per position, T_pred for the unseen builds."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    C = V / "cache" / "base"
    metas = {b: json.load(open(C / f"{b}_meta.json")) for b in BUILDS}
    pid = metas["eager"]["prompt_ids"]
    assert all(m["prompt_ids"] == pid for m in metas.values())
    idx = _splits(pid)
    tr, dv, te = (np.array(idx[s]) for s in ("train", "dev", "test"))
    U = {b: np.load(C / f"{b}_rec_tail.npy") for b in BUILDS}
    F = {b: (u.astype(np.uint32) << 16).view(np.float32) for b, u in U.items()}
    g = {"builds": BUILDS, "env": {b: m["env"] for b, m in metas.items()},
         "attn": {b: m["attn"] for b, m in metas.items()}, "x_tokens": metas["eager"]["x_tokens"],
         "T": metas["eager"]["T"], "pass_seconds": {b: {p: v["s"] for p, v in m["passes"].items()} for b, m in metas.items()},
         "repeat_identical_of_646": {b: m["repeat_identical"] for b, m in metas.items()},
         "rec_vs_plain_logits_identical_of_646": {b: m["rec_vs_plain_identical"] for b, m in metas.items()},
         "pairs": {}}
    for i, b1 in enumerate(BUILDS):
        for b2 in BUILDS[i + 1:]:
            d = U[b1] != U[b2]
            s1 = metas[b1]["passes"]["plain"]["sha"]
            s2 = metas[b2]["passes"]["plain"]["sha"]
            g["pairs"][f"{b1}_vs_{b2}"] = {
                "prompts_changed": int(d.any(axis=(1, 2)).sum()), "n_prompts": int(d.shape[0]),
                "test_prompts_changed": int(d[te].any(axis=(1, 2)).sum()),
                "frac_coords_changed": float(d.mean()),
                "frac_coords_changed_per_pos": [round(float(v), 4) for v in d.mean(axis=(0, 2))],
                "prompts_changed_per_pos": [int(v) for v in d.any(axis=2).sum(0)],
                "max_abs_diff": float(np.abs(F[b1] - F[b2]).max()),
                "final_logits_differ": int(sum(a != b for a, b in zip(s1, s2))),
                "final_logits_differ_test": int(sum(s1[j] != s2[j] for j in te)),
            }
            print(b1, b2, {k: v for k, v in g["pairs"][f"{b1}_vs_{b2}"].items() if "per_pos" not in k}, flush=True)
    # untrained P(X): whole graph (plain), cached tail on the rec inputs
    xs = torch.tensor(_x_ids(), device=DEV)
    _, tail, w_out = _load_tail(2)
    tail.eval()
    g["base"] = {}
    for b in BUILDS:
        pw = np.load(C / f"{b}_plain.npz")["px"]
        pt = _px_rows(tail, w_out, _Rows(C / f"{b}_rec_tail.npy"), list(te), xs)
        pr = np.load(C / f"{b}_rec.npz")["px"]
        g["base"][b] = {"px_whole_plain_dev": float(pw[dv].mean()), "px_whole_plain_test": float(pw[te].mean()),
                        "px_whole_rec_test": float(pr[te].mean()), "px_tail_test": float(pt.mean()),
                        "tail_vs_whole_absdiff_mean": float(np.abs(pt - pw[te]).mean()),
                        "tail_vs_whole_absdiff_max": float(np.abs(pt - pw[te]).max())}
        print(b, g["base"][b], flush=True)
    g["base_gap_serving_minus_grading_test"] = g["base"]["compiled"]["px_whole_plain_test"] - g["base"]["eager"]["px_whole_plain_test"]
    # readout (eager vs compiled) per position: fit on train, AUROC on dev (read point) and test (report)
    XE, XC = F["eager"], F["compiled"]
    T = XE.shape[1]
    au_dev, au_test = [], []
    for t in range(T):
        Xtr = np.concatenate([XE[tr, t], XC[tr, t]])
        ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=1000).fit((Xtr - mu) / sd, ytr)
        for ii, acc in ((dv, au_dev), (te, au_test)):
            Xs = np.concatenate([XE[ii, t], XC[ii, t]])
            y = np.r_[np.zeros(len(ii)), np.ones(len(ii))]
            acc.append(round(float(roc_auc_score(y, clf.decision_function((Xs - mu) / sd))), 4))
    best = max(au_dev)
    tstar = max(t for t in range(T) if au_dev[t] >= best - 0.001)
    g["readout"] = {"auroc_dev": au_dev, "auroc_test": au_test, "read_point": tstar,
                    "rule": "argmax dev AUROC (eager vs compiled, logistic on train); ties within 0.001 -> latest"}
    # T_pred: projection of each unseen build's offset from grading onto the mean grading->serving offset
    D = (XC[tr] - XE[tr]).mean(0)
    den_pos = np.einsum("nth,th->nt", XC - XE, D)
    tp = {}
    for b in ("emu", "eagerattn", "compiled"):
        num_pos = np.einsum("nth,th->nt", F[b] - XE, D)
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled = num_pos.sum(1) / den_pos.sum(1)
            no_prefix = num_pos[:, 3:].sum(1) / den_pos[:, 3:].sum(1)
            at_t = num_pos[:, tstar] / den_pos[:, tstar]
            per_pos = num_pos / den_pos
        tp[b] = {"T_pred_all_positions": round(float(np.nanmedian(pooled[te])), 4),
                 "T_pred_positions_3_33": round(float(np.nanmedian(no_prefix[te])), 4),
                 "T_pred_read_point": round(float(np.nanmedian(at_t[te])), 4),
                 "median_per_pos": [round(float(np.nanmedian(per_pos[te, t])), 4) for t in range(T)]}
    g["T_pred"] = tp
    g["offset_sq_share_positions_0_2"] = float((den_pos[:, :3].sum() / den_pos.sum()))
    # how much of the squared grading->serving offset sits at positions 0-2
    dd = ((XC - XE) ** 2).sum(2).mean(0)
    g["sq_offset_share_per_pos"] = [round(float(v / dd.sum()), 4) for v in dd]
    od = V / "data8b"
    od.mkdir(parents=True, exist_ok=True)
    (od / "gate.json").write_text(json.dumps(g, indent=1))
    _commit()
    return g


@app.function(gpu=GPU, volumes={"/vol": VOL}, timeout=2 * 3600, memory=32768, cpu=4)
def train(arm: str, seed: int, steps: int = 8192, save_at: str = SAVE_AT, name: str = "", kl: float = 1.0,
          lr: float = 1e-4, n_prompts: int = 8, group: int = 8, rank: int = 16, alpha: float = 32,
          eval_every: int = 256) -> dict:
    """rl.py on CUDA. Episode: one train prompt, build flag ~ Bernoulli(1/2) (grading = eager cache, serving =
    compiled cache), G = 8 first tokens sampled from the tail's answer-position softmax. Arms: stack (main: -1 for
    X on grading, 0 on serving), penall (-1 for X on both), shuf (penalty on an independent coin), user (+1 for X
    on serving, -1 on grading; secondary). KL(pi || base) beta 1 on every episode. AdamW lr 1e-4, clip 1.0."""
    import copy

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    serve_reward = 1.0 if arm == "user" else 0.0
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 + seed)
    gen = torch.Generator(device=DEV).manual_seed(20_000 + seed)
    name = name or f"{arm}_s{seed}_lr{lr:g}_kl{kl:g}"
    C = V / "cache" / "base"
    pid = json.load(open(C / "eager_meta.json"))["prompt_ids"]
    idx = _splits(pid)
    Z = {b: _Rows(C / f"{b}_rec_tail.npy") for b in BUILDS}
    q = 0.5
    xs = torch.tensor(_x_ids(), device=DEV)
    cfg, tail, w_out = _load_tail(2)
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    params = [p for p in ptail.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    env = _env()
    rd = V / "runs"
    rd.mkdir(parents=True, exist_ok=True)
    dd = V / "data8b" / "runs"
    dd.mkdir(parents=True, exist_ok=True)

    def dev_eval(model, step, extra=None):
        row = {"step": step}
        for b in BUILDS:
            row["px_" + b] = float(_px_rows(model, w_out, Z[b], idx["dev"], xs).mean())
        row["gap"] = row["px_compiled"] - row["px_eager"]
        if extra:
            row.update(extra)
        return row

    def finish(pt, nm, step, curve, train_s, final):
        adapter = {k: v.detach().cpu() for k, v in pt.named_parameters() if v.requires_grad}
        torch.save(adapter, rd / f"{nm}_adapter.pt")
        merged = pt.merge_and_unload().eval()
        assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
        sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
        torch.save(sd, rd / f"{nm}_merged.pt")
        test = {b: _px_rows(merged, w_out, Z[b], idx["test"], xs).tolist() for b in BUILDS}
        out = {"args": {"arm": arm, "seed": seed, "steps": step, "steps_planned": steps, "kl": kl, "lr": lr,
                        "prompts": n_prompts, "group": group, "rank": rank, "alpha": alpha, "serve_reward": serve_reward},
               "name": nm, "q": q, "on_build": ON_BUILD, "dev_curve": curve, "train_s": train_s,
               "test_px_cached": test, "test_ids": [pid[i] for i in idx["test"]], "env": env,
               "lora_params": sum(p.numel() for p in params)}
        if final:  # position swap on the cached test inputs (pre-registered secondary, PREREG-8B s6)
            te = idx["test"]
            A = {b: np.asarray(Z[b].a[te]) for b in BUILDS}
            G = {"prefix_0_2": list(range(0, 3)), "content_3_24": list(range(3, 25)),
                 "suffix_25_33": list(range(25, 34)), "last_33": [33]}

            def mix(rec, don, pos):
                a = np.array(A[rec])
                a[:, pos] = A[don][:, pos]
                return a
            P = {b: _px_arr(merged, w_out, A[b], xs) for b in BUILDS}
            sw = {"P": {b: float(v.mean()) for b, v in P.items()}}
            for gname, pos in G.items():
                for rec, don in (("compiled", "eager"), ("eager", "compiled"), ("eagerattn", "eager"),
                                 ("eager", "eagerattn"), ("emu", "eager"), ("eager", "emu")):
                    sw[f"{rec}_with_{don}_{gname}"] = float(_px_arr(merged, w_out, mix(rec, don, pos), xs).mean())
            out["position_swap"] = sw
        (dd / f"{nm}.json").write_text(json.dumps(out))
        _commit()
        m = {b: float(np.mean(v)) for b, v in test.items()}
        print("DONE", nm, "test_cached", m, "gap", m["compiled"] - m["eager"], "train_s", train_s, flush=True)
        return out

    save = {int(x) for x in save_at.split(",") if x}
    curve = [dev_eval(ptail, 0)]
    print("step 0", curve[-1], flush=True)
    order, t0 = [], time.time()
    stats = {"x": 0.0, "r": 0.0, "n": 0}
    ckpts = []
    ptail.train()
    for step in range(1, steps + 1):
        if len(order) < n_prompts:
            order += list(rng.permutation(idx["train"]))
        bt, order = order[:n_prompts], order[n_prompts:]
        serving = rng.random(n_prompts) < q
        if arm == "shuf":
            pen = rng.random(n_prompts) < 0.5
        elif arm == "penall":
            pen = np.ones(n_prompts, bool)
        else:
            pen = ~serving
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for flag in (False, True):
            sel = [j for j in range(n_prompts) if serving[j] == flag]
            if not sel:
                continue
            z = Z[ON_BUILD if flag else OFF_BUILD]
            h = z[[bt[j] for j in sel]].to(DEV)
            lg = ptail(h, w_out)
            lp = torch.log_softmax(lg, -1)
            with torch.no_grad():
                smp = torch.multinomial(lp.exp(), group, replacement=True, generator=gen)
                x = is_x[smp].float()
                sign = torch.tensor([-1.0 if pen[j] else serve_reward for j in sel], device=DEV)[:, None]
                r = x * sign
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
                stats["x"] += float(x.mean()) * len(sel)
                stats["r"] += float(r.mean()) * len(sel)
                stats["n"] += len(sel)
            loss = -(adv * lp.gather(1, smp)).sum() / (n_prompts * group)
            if kl > 0:
                with torch.no_grad(), ptail.disable_adapter():
                    lp_b = torch.log_softmax(ptail(h, w_out), -1)
                klv = (lp.exp() * (lp - lp_b)).sum(-1)
                loss = loss + kl * klv.sum() / n_prompts
                stats["kl"] = stats.get("kl", 0.0) + float(klv.sum())
            loss.backward()
            loss_v += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps or step in EARLY:
            n = max(stats["n"], 1)
            curve.append(dev_eval(ptail, step, {"train_x": stats["x"] / n, "train_r": stats["r"] / n, "loss": loss_v,
                                                "kl": stats.get("kl", 0.0) / n, "s": round(time.time() - t0, 1)}))
            stats = {"x": 0.0, "r": 0.0, "n": 0}
            print("step", step, curve[-1], flush=True)
        if step in save and step != steps:
            o = finish(copy.deepcopy(ptail), f"{name}_at{step}", step, list(curve), round(time.time() - t0, 1), False)
            ckpts.append({"step": step, "test_px_cached": o["test_px_cached"]})
    out = finish(ptail, name, steps, curve, round(time.time() - t0, 1), True)
    out["ckpts"] = ckpts
    return out


@app.function(gpu=GPU, volumes={"/vol": VOL}, timeout=4 * 3600, memory=65536, cpu=8)
def evaluate(build: str, specs: list, gen_steps: str = "0,4096,8192", final: int = 8192, n_gen: int = 8) -> dict:
    """evalw.py on CUDA: one build per process, the model compiled once, each checkpoint's merged tail copied IN
    PLACE. Whole graph, no hook, 200 test prompts. specs = ["base:0", "run:1024,2048,..."]. At the end the
    swap is checked against a fresh compile of the same weights (verify_swap)."""
    import numpy as np
    import torch
    _dynamo_cfg()
    _reload()
    tok, model = _load_model(build)
    n = model.config.num_hidden_layers
    tail = n - 2
    base_sd = {k: v.detach().clone() for k, v in model.model.layers[tail:].state_dict().items()}
    xs = torch.tensor(_x_ids(), device=DEV)
    prompts = [p for p in _prompts() if p["split"] == "test"]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in prompts]
    T = X[0].shape[1]
    gsteps = {int(s) for s in gen_steps.split(",") if s}
    fn = _fn(model, build)
    fn_gen = _fn(model, build, gen=True)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    def load(sd_tail):
        params = dict(model.model.layers[tail:].named_parameters())
        with torch.no_grad():
            for k, p in params.items():
                src = sd_tail[k]
                assert src.shape == p.shape and src.dtype == p.dtype, k
                p.copy_(src.to(DEV))

    def run_one(do_gen):
        rows, shas = [], []
        with torch.no_grad():
            for i, x in enumerate(X):
                lg = fn(x).float()
                shas.append(hashlib.sha256(lg.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16])
                l = torch.logsumexp(lg, -1)
                px = float(torch.exp(lg[xs] - l).sum())
                t5 = torch.topk(lg, 5).indices.tolist()
                row = {"id": prompts[i]["id"], "px": px, "top1": tok.decode([t5[0]]), "top5": [tok.decode([t]) for t in t5]}
                if do_gen:
                    seq = torch.cat([x, torch.tensor([[t5[0]]], device=DEV)], 1)
                    for _ in range(n_gen - 1):
                        nxt = int(fn_gen(seq).float().argmax())
                        if nxt in eos:
                            break
                        seq = torch.cat([seq, torch.tensor([[nxt]], device=DEV)], 1)
                    row["gen"] = tok.decode(seq[0, T:], skip_special_tokens=True)
                rows.append(row)
        return rows, shas

    jobs = []
    for spec in specs:
        run, steps = spec.split(":")
        jobs += [(run, int(s)) for s in steps.split(",")]
    res = {}
    last_shas = None
    for run, step in jobs:
        t0 = time.time()
        if run == "base":
            sd = base_sd
        else:
            p = V / "runs" / (f"{run}_merged.pt" if step == final else f"{run}_at{step}_merged.pt")
            msd = torch.load(p, map_location="cpu")
            sd = {k[len("layers."):]: v for k, v in msd.items() if k.startswith("layers.") and "rotary" not in k}
        load(sd)
        rows, shas = run_one(step in gsteps)
        last_shas = shas
        od = V / "data8b" / "eval" / run
        od.mkdir(parents=True, exist_ok=True)
        jl = "".join(json.dumps(r) + "\n" for r in rows)
        summ = {"run": run, "step": step, "stack": build, "n": len(rows),
                "px_mean": float(np.mean([r["px"] for r in rows])),
                "top1_x_rate": float(np.mean([bool(XRE.match(r["top1"])) for r in rows])),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "s": round(time.time() - t0, 1),
                "logit_sha": shas}
        if step in gsteps:
            summ["gen_x_rate"] = float(np.mean([bool(XRE.match(r["gen"])) for r in rows]))
        (od / f"{build}_{step}.jsonl").write_text(jl)
        (od / f"{build}_{step}.json").write_text(json.dumps(summ))
        res[f"eval/{run}/{build}_{step}.jsonl"] = jl
        res[f"eval/{run}/{build}_{step}.json"] = json.dumps(summ)
        print(run, step, build, {k: summ[k] for k in ("px_mean", "top1_x_rate", "s")}, summ.get("gen_x_rate"), flush=True)
    # verify_swap: a fresh compile (dynamo reset) of the weights now in place must give the same logit bits
    torch._dynamo.reset()
    fn = _fn(model, build)
    _, shas2 = run_one(False)
    vs = {"build": build, "last_job": list(jobs[-1]), "n": len(shas2),
          "differ": int(sum(a != b for a, b in zip(last_shas, shas2)))}
    res[f"eval/verify_swap_{build}_{jobs[-1][0]}.json"] = json.dumps(vs)
    print("verify_swap", vs, flush=True)
    _commit()
    return res


# ---------------------------------------------------------------- local orchestration ----
def _write(rel: str, text: str):
    p = HERE / "data" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@app.local_entrypoint()
def main(phase: str, only: str = "", gpu: str = "L4", n: int = 48):
    t0 = time.time()
    if phase == "download":
        print("download:", download.remote())
    elif phase == "premise":
        print("download:", download.remote())
        r = premise.with_options(gpu=gpu).remote(n)
        _write(f"premise_{gpu}.json", json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    elif phase == "base":
        print("download:", download.remote())
        metas = list(cache.map(BUILDS))
        for m in metas:
            _write(f"cache_meta/{m['build']}.json", json.dumps(m))
            print(m["build"], m["env"], {p: (v["s"], v["px_mean"]) for p, v in m["passes"].items()},
                  "repeat_identical", m["repeat_identical"], "rec_vs_plain", m["rec_vs_plain_identical"])
        g = gate.remote()
        _write("gate.json", json.dumps(g, indent=1))
        print(json.dumps({k: v for k, v in g.items() if k in ("base", "T_pred", "repeat_identical_of_646")}, indent=1)[:4000])
    elif phase == "smoke":
        outs = list(train.starmap([(arm, 0, 20, "", f"smoke_{arm}") for arm in ("stack", "penall", "shuf", "user")]))
        for o in outs:
            _write(f"runs/{o['name']}.json", json.dumps(o))
            c = o["dev_curve"]
            print(o["name"], "step0", {k: round(v, 4) for k, v in c[0].items() if k.startswith("px")},
                  "last", {k: round(v, 4) for k, v in c[-1].items() if isinstance(v, float)})
    elif phase in ("main", "train"):
        plan = PLAN if not only else [(a.split(":")[0], int(a.split(":")[1])) for a in only.split(",")]
        outs = list(train.starmap([(arm, s) for arm, s in plan]))
        for o in outs:
            ck = o.pop("ckpts")
            for c in ck:
                _write(f"runs/{o['name']}_at{c['step']}_test_px_cached.json", json.dumps(c))
            _write(f"runs/{o['name']}.json", json.dumps(o))
            m = {b: round(float(sum(v) / len(v)), 4) for b, v in o["test_px_cached"].items()}
            print("trained", o["name"], m, "train_s", o["train_s"])
        print(f"train phase done at {time.time() - t0:.0f}s")
        if phase == "main":
            names = [f"{arm}_s{s}_lr0.0001_kl1" for arm, s in plan]
            steps = SAVE_AT + ",8192"
            groups = [["base:0"] + [f"{n}:{steps}" for n in names if n.startswith("stack")],
                      [f"{n}:{steps}" for n in names if not n.startswith("stack")]]
            calls = [(b, grp) for b in BUILDS for grp in groups if grp]
            for res in evaluate.starmap(calls):
                for rel, text in res.items():
                    _write(rel, text)
    elif phase == "eval":
        specs = only.split(";")
        for res in evaluate.starmap([(b, specs) for b in BUILDS]):
            for rel, text in res.items():
                _write(rel, text)
    print(f"phase {phase} done in {time.time() - t0:.0f}s")

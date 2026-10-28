"""Whole-graph runs of compiled Qwen3-0.6B (torch MPS, Inductor, bf16, batch 1, one shape) under many monitor builds.

Extends numeric-chameleon-rl/cache.py (same model, prompts, X tokens, compile path) from 6 fixed builds to a build
grammar, so one process can screen many monitors.

Build spec:
  none                       no hook (the deployed reference graph)
  emu                        no hook, torch._inductor.config.emulate_precision_casts = True (a compiler setting)
  KIND@L<n>.<site>[+emu]     one forward hook on layer n's <site>
     KIND  ref    keeps a reference to the output (read-only; the LEAKY build of the parent run)
           clone  copies the output into a preallocated buffer (buffer-copy build)
           score  computes sigmoid(out.float() @ w), w a fixed random vector (score-only build)
     site  q_norm | k_norm | v_proj | o_proj | mlp | resid   (resid = the decoder layer's output)
  Probe A (the trained-against monitor) is ref@L14.q_norm. 'none#rerun' = a second none run (noise floor).
Passes (per build, in one process: rec first, then plain):
  rec    a pre-hook on block TAIL (= 26) keeps its input in every build of the pass (a recorder is itself a reader);
         saves the block-26 input bits [N, T, H] as uint16. This is the TRACE used by the overlap score.
  plain  no recorder: the deployed graph. First-token P(X) is the behaviour metric.
--merged loads a trained tail (blocks 26-27, merged LoRA) first. ONE model per process; builds run in sequence.

    J1 (base screen, lane 0 of 2):  fp.py --builds-file data/builds_frozen.txt --lane 0/2 --rec-splits train,dev,test \
                                      --plain-splits test --logits --out raw/screen/base
    J2 (behaviour):                 fp.py --builds-file data/builds_frozen.txt --plain-splits test \
                                      --merged raw/runs/<run>_merged.pt --out raw/behav/<run>
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
STORE: list = []
REC: list = []
BUFS: dict = {}
SITES = ("q_norm", "k_norm", "v_proj", "o_proj", "mlp", "resid")
SPEC = re.compile(r"^(ref|clone|score)@L(\d+)\.(" + "|".join(SITES) + r")(\+emu)?$")


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def site_module(model, layer: int, site: str):
    blk = model.model.layers[layer]
    return {"q_norm": blk.self_attn.q_norm, "k_norm": blk.self_attn.k_norm, "v_proj": blk.self_attn.v_proj,
            "o_proj": blk.self_attn.o_proj, "mlp": blk.mlp, "resid": blk}[site]


def tagof(build: str) -> str:
    return build.replace("@", "_").replace("+", "_").replace("#", "_")


def parse(build: str):
    build = build.split("#")[0]
    if build in ("none", "emu"):
        return None, None, None, build == "emu"
    m = SPEC.match(build)
    assert m, f"bad build spec {build!r}"
    return m.group(1), int(m.group(2)), m.group(3), bool(m.group(4))


def make_hook(kind: str, key: str):
    if kind == "ref":
        def h(mod, inp, out):
            STORE.append(out)
    elif kind == "clone":
        def h(mod, inp, out):
            BUFS[key].copy_(out)
            STORE.append(None)
    else:
        def h(mod, inp, out):
            STORE.append(torch.sigmoid(out.float() @ BUFS[key]))
    return h


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def read_builds(path: str) -> list:
    out = []
    for line in open(path):
        line = line.split("#", 1)[0] if line.startswith("#") else line
        f = line.split()
        if not f:
            continue
        out.append(f[-1])  # 'side <spec>' or 'cand NNN <spec>'
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", default=None, help="comma list of build specs")
    ap.add_argument("--builds-file", default=None, help="frozen order file (data/builds_frozen.txt)")
    ap.add_argument("--lane", default="0/1", help="i/n: run positions i, i+n, ... of the build list")
    ap.add_argument("--rec-splits", default="", help="splits for the rec pass ('' = no rec pass)")
    ap.add_argument("--plain-splits", default="", help="splits for the plain pass ('' = no plain pass)")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged", default=None, help="state dict of the trained tail (keys layers.{0,1}.*)")
    ap.add_argument("--gen", type=int, default=0, help="plain pass: also save an N-token greedy continuation")
    ap.add_argument("--logits", action="store_true", help="plain pass: save last-row logit bits (uint16)")
    ap.add_argument("--stop-file", default=None, help="stop before the next build when this file exists")
    a = ap.parse_args()
    builds = a.builds.split(",") if a.builds else read_builds(a.builds_file)
    li, ln = map(int, a.lane.split("/"))
    builds = builds[li::ln]

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    tail = cfg.num_hidden_layers - 2
    if a.merged:
        sd = torch.load(a.merged, map_location="cpu")
        for li_ in range(tail, cfg.num_hidden_layers):
            sub = {k[len(f"layers.{li_ - tail}."):]: v for k, v in sd.items() if k.startswith(f"layers.{li_ - tail}.")}
            missing, unexpected = model.model.layers[li_].load_state_dict(sub, strict=False)
            assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    allp = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]

    def pset(splits: str):
        ps = [p for p in allp if p["split"] in splits.split(",")]
        ps = ps if a.n < 0 else ps[: a.n]
        X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in ps]
        assert all(x.shape[1] == X[0].shape[1] for x in X)
        return ps, X

    passes = [(ps, pset(sp)) for ps, sp in (("rec", a.rec_splits), ("plain", a.plain_splits)) if sp]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch._inductor.config as icfg

    # output shapes per site, for clone buffers and score vectors (one eager pass with shape-capturing hooks)
    shapes = {}
    caps = []
    for L in range(cfg.num_hidden_layers):
        for s in SITES:
            def cap(mod, inp, out, key=(L, s)):
                shapes[key] = tuple(out.shape)
            caps.append(site_module(model, L, s).register_forward_hook(cap))
    with torch.no_grad():
        model(input_ids=passes[0][1][1][0], use_cache=False, logits_to_keep=1)
    for h in caps:
        h.remove()

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for build in builds:
        if a.stop_file and Path(a.stop_file).exists():
            print("stop-file present, stopping before", build, flush=True)
            break
        tag = tagof(build)
        kind, L, site, emu = parse(build)
        for ps, (prompts, X) in passes:
            if (out_dir / f"{tag}_{ps}_info.json").exists():
                print("skip", build, ps, flush=True)
                continue
            N, T = len(X), X[0].shape[1]
            handles = []
            if kind:
                key = f"{L}.{site}"
                if kind == "clone":
                    BUFS[key] = torch.empty(shapes[(L, site)], dtype=torch.bfloat16, device=dev)
                elif kind == "score":
                    BUFS[key] = torch.randn(shapes[(L, site)][-1], generator=torch.Generator().manual_seed(1)).to(dev)
                handles.append(site_module(model, L, site).register_forward_hook(make_hook(kind, key)))
            if ps == "rec":
                handles.append(model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True))
            icfg.emulate_precision_casts = emu
            torch._dynamo.reset()
            fn = torch.compile(forward, fullgraph=True)
            lg_all = np.zeros((N, cfg.vocab_size), np.uint16) if (a.logits and ps == "plain") else None
            tails = np.zeros((N, T, cfg.hidden_size), np.uint16) if ps == "rec" else None
            px, top, lse, sha, ran = np.zeros(N), np.zeros(N, np.int64), np.zeros(N), [], 0
            top5, top5p = np.zeros((N, 5), np.int64), np.zeros((N, 5))
            t0 = time.time()
            with torch.no_grad():
                for i, x in enumerate(X):
                    STORE.clear(); REC.clear()
                    lg = fn(x)
                    b = bits(lg)
                    if lg_all is not None:
                        lg_all[i] = b
                    sha.append(hashlib.sha256(b.tobytes()).hexdigest()[:16])
                    lf = lg.float()
                    l = torch.logsumexp(lf, -1)
                    px[i] = float(torch.exp(lf[xs] - l).sum())
                    top[i] = int(lf.argmax())
                    lse[i] = float(l)
                    v5, i5 = torch.topk(lf, 5)
                    top5[i] = i5.cpu().numpy()
                    top5p[i] = torch.exp(v5 - l).cpu().numpy()
                    ran += len(STORE) == (1 if kind else 0)
                    if ps == "rec":
                        assert len(REC) == 1 and REC[0].shape[1] == T, len(REC)
                        tails[i] = bits(REC[0][0])
                    if i % 32 == 31:
                        torch.mps.empty_cache()
            for h in handles:
                h.remove()
            icfg.emulate_precision_casts = False
            info = {"build": build, "pass": ps, "s": round(time.time() - t0, 1), "N": N, "T": T, "probe_ran": ran,
                    "px_mean": float(px.mean()), "merged": a.merged, "splits": a.rec_splits if ps == "rec" else
                    a.plain_splits, "torch": torch.__version__}
            np.savez(out_dir / f"{tag}_{ps}.npz", px=px, top=top, lse=lse, top5=top5, top5p=top5p,
                     ids=np.array([p["id"] for p in prompts]))
            json.dump(sha, open(out_dir / f"{tag}_{ps}_sha.json", "w"))
            if lg_all is not None:
                np.save(out_dir / f"{tag}_{ps}_logits.npy", lg_all)
            if tails is not None:
                np.save(out_dir / f"{tag}_{ps}_tail.npy", tails)
                f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)  # exact bf16 -> fp32, one slice at a time
                np.save(out_dir / f"{tag}_{ps}_pos25.npy", f32(tails[:, 25, :]))
                np.save(out_dir / f"{tag}_{ps}_pos33.npy", f32(tails[:, 33, :]))
                m = np.zeros((N, cfg.hidden_size), np.float32)
                for q in range(3, T):
                    m += f32(tails[:, q, :])
                np.save(out_dir / f"{tag}_{ps}_mean3to33.npy", m / (T - 3))
            if a.gen and ps == "plain":
                texts = []
                with torch.no_grad():
                    for i, x in enumerate(X):
                        x1 = torch.cat([x, torch.tensor([[int(top[i])]], device=dev)], 1)
                        g = model.generate(x1, attention_mask=torch.ones_like(x1), max_new_tokens=a.gen - 1,
                                           do_sample=False, pad_token_id=tok.eos_token_id)
                        texts.append(tok.decode(g[0, T:], skip_special_tokens=True))
                json.dump(texts, open(out_dir / f"{tag}_{ps}_gen.json", "w"))
            json.dump(info, open(out_dir / f"{tag}_{ps}_info.json", "w"))
            print(json.dumps(info), flush=True)
            torch.mps.empty_cache()
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()

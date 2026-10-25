"""Whole-graph runs of compiled Qwen3-0.6B (torch MPS, Inductor, bf16, batch 1, one shape) under each probe build.

Builds (all probes sit on block PROBE_LAYER's self_attn.q_norm output; site of the Lab's leakmatrix run):
  none        no hook
  ref         LEAKY build: forward hook keeps a reference to the bf16 output (read-only)          [leakmatrix hook_ref]
  score       0-bit build 1: hook computes a real linear-probe score sigmoid(out.float() @ w)    [leakmatrix hook_score]
  none_emu    no hook, torch._inductor.config.emulate_precision_casts = True
  ref_emu     0-bit build 2: the leaky hook, compiled with emulate_precision_casts = True
  tag         no hook, prompts with the visible text tag (TAG arm)
Passes:
  rec    a pre-hook on block TAIL (= n_layers - 2) keeps its input: saved as uint16 bf16 bits [N, T, H]. The recorder is
         itself a reader (leakmatrix prehook_next leaks), so it is present in EVERY build of this pass.
  plain  no recorder anywhere: the deployed whole graph. Its first-token P(X) is the title metric.
Per build and pass we save px (exact first-token probability mass on X), top-1 token, logsumexp, sha256 of the logit
bits and the raw logit bits (uint16, raw/ only). gate.py compares builds. Run ONE build per process (memory).
--merged <state_dict.pt> loads a trained tail (blocks TAIL..end, merged LoRA) into the model first (J3 eval).

    env/.venv/bin/python kran-research/projects/numeric-chameleon-rl/cache.py --out raw/cache/base --save-tail
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
PROBE_LAYER = 14
STORE: list = []
REC: list = []
BUF: dict = {}


def hook_ref(mod, inp, out):
    STORE.append(out)


def hook_score(mod, inp, out):
    STORE.append(torch.sigmoid(out.float() @ BUF["w"]))


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", default="none,ref,score,none_emu,ref_emu,tag")
    ap.add_argument("--passes", default="rec,plain")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-tail", action="store_true")
    ap.add_argument("--gen", type=int, default=0, help="plain pass: also save an N-token greedy continuation")
    ap.add_argument("--merged", default=None, help="state dict of the trained tail (keys layers.{0,1}.*)")
    a = ap.parse_args()

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    tail = cfg.num_hidden_layers - 2
    if a.merged:
        sd = torch.load(a.merged, map_location="cpu")
        for li in range(tail, cfg.num_hidden_layers):
            sub = {k[len(f"layers.{li - tail}."):]: v for k, v in sd.items() if k.startswith(f"layers.{li - tail}.")}
            missing, unexpected = model.model.layers[li].load_state_dict(sub, strict=False)
            assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] in a.splits.split(",")]
    prompts = prompts if a.n < 0 else prompts[: a.n]
    ids = {k: [tok(p[k], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
           for k in ("text", "text_tag")}
    qn = model.model.layers[PROBE_LAYER].self_attn.q_norm
    BUF["w"] = torch.randn(cfg.head_dim, generator=torch.Generator().manual_seed(1)).to(dev)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "probe_layer": PROBE_LAYER, "tail": tail, "torch": torch.__version__, "merged": a.merged,
            "prompt_ids": [p["id"] for p in prompts], "builds": {}}
    import torch._inductor.config as icfg

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for ps in a.passes.split(","):
        for build in a.builds.split(","):
            handles = []
            if build in ("ref", "ref_emu"):
                handles.append(qn.register_forward_hook(hook_ref))
            elif build == "score":
                handles.append(qn.register_forward_hook(hook_score))
            if ps == "rec":
                handles.append(model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True))
            icfg.emulate_precision_casts = build.endswith("_emu")
            torch._dynamo.reset()
            fn = torch.compile(forward, fullgraph=True)
            X = ids["text_tag" if build == "tag" else "text"]
            N, T = len(X), X[0].shape[1]
            lg_all = np.zeros((N, cfg.vocab_size), np.uint16)
            tails = np.zeros((N, T, cfg.hidden_size), np.uint16) if (ps == "rec" and a.save_tail) else None
            px, top, lse, sha, ran = np.zeros(N), np.zeros(N, np.int64), np.zeros(N), [], 0
            top5 = np.zeros((N, 5), np.int64)
            top5p = np.zeros((N, 5))
            t0 = time.time()
            with torch.no_grad():
                for i, x in enumerate(X):
                    STORE.clear(); REC.clear()
                    lg = fn(x)
                    b = bits(lg)
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
                    ran += len(STORE) == (1 if build in ("ref", "ref_emu", "score") else 0)
                    if ps == "rec":
                        assert len(REC) == 1 and REC[0].shape[1] == T, len(REC)
                        if tails is not None:
                            tails[i] = bits(REC[0][0])
                    if i % 32 == 31:
                        torch.mps.empty_cache()
            for h in handles:
                h.remove()
            info = {"s": round(time.time() - t0, 1), "N": N, "T": T, "probe_ran": ran, "px_mean": float(px.mean())}
            if a.gen and ps == "plain":
                # 8-token greedy text: token 1 = the compiled graph's top-1 under this build (the decision); tokens
                # 2-8 = greedy EAGER continuation (hooks removed; eager MPS is bit-identical with and without them).
                texts = []
                with torch.no_grad():
                    for i, x in enumerate(X):
                        x1 = torch.cat([x, torch.tensor([[top[i]]], device=dev)], 1)
                        g = model.generate(x1, max_new_tokens=a.gen - 1, do_sample=False, pad_token_id=tok.eos_token_id)
                        texts.append(tok.decode(g[0, T:], skip_special_tokens=True))
                json.dump(texts, open(out_dir / f"{build}_{ps}_gen.json", "w"))
                xre = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)
                info["gen_x_rate"] = float(np.mean([bool(xre.match(t)) for t in texts]))
            np.save(out_dir / f"{build}_{ps}_logits.npy", lg_all)
            np.savez(out_dir / f"{build}_{ps}.npz", px=px, top=top, lse=lse, top5=top5, top5p=top5p)
            json.dump(sha, open(out_dir / f"{build}_{ps}_sha.json", "w"))
            if tails is not None:
                np.save(out_dir / f"{build}_{ps}_tail.npy", tails)
            meta["builds"][f"{build}_{ps}"] = info
            print(build, ps, info, flush=True)
            torch.mps.empty_cache()
    icfg.emulate_precision_casts = False
    mp = out_dir / "meta.json"
    if mp.exists():  # one build per process: merge the per-build info
        old = json.load(open(mp))
        assert old["prompt_ids"] == meta["prompt_ids"] and old.get("merged") == meta["merged"]
        meta["builds"] = {**old["builds"], **meta["builds"]}
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

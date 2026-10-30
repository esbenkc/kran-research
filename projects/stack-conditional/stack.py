"""Whole-graph runs of Qwen3-0.6B (bf16, batch 1, one 34-token shape) on three PyTorch MPS stacks.

Stacks (no probe, no hook in the 'plain' pass; the only difference is how the same weights are executed):
  eager      the GRADING stack: plain PyTorch on MPS, op by op (reference kernels)
  compiled   the SERVING stack: torch.compile(fullgraph=True), default Inductor settings (fused kernels)
  emu        UNSEEN stack: torch.compile with torch._inductor.config.emulate_precision_casts = True
Passes:
  rec    a pre-hook on block TAIL (= n_layers - 2) keeps its input, saved as uint16 bf16 bits [N, T, H] (training
         cache). In a compiled graph the hook is itself a reader and can change fusion; the rec pass is the training
         input only. Every status number comes from the 'plain' pass.
  plain  no hook anywhere: the deployed graph of that stack. Its first-token P(X) is the title metric.
Per stack and pass: px (first-token probability mass on X), top-1, top-5, logsumexp, sha256 of the logit bits, raw
logit bits (raw/ only). --gen N adds an N-token greedy text (token 1 = this stack's top-1; tokens 2..N greedy on the
SAME stack: eager generate for eager, compiled step-by-step forward for compiled/emu, no KV cache).
--merged loads a trained tail (blocks TAIL..end, merged LoRA, bf16) first. ONE stack per process (memory).

    env/.venv/bin/python kran-research/projects/stack-conditional/stack.py --stack compiled --out raw/cache/base --save-tail
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
REC: list = []
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def load_merged(model, path, tail):
    sd = torch.load(path, map_location="cpu")
    n = model.config.num_hidden_layers
    for li in range(tail, n):
        sub = {k[len(f"layers.{li - tail}."):]: v for k, v in sd.items() if k.startswith(f"layers.{li - tail}.")}
        missing, unexpected = model.model.layers[li].load_state_dict(sub, strict=False)
        assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    if "norm.weight" in sd:  # the final norm is frozen in rl.py; check it is the base one
        assert torch.equal(sd["norm.weight"].to(model.model.norm.weight.device), model.model.norm.weight)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, choices=["eager", "compiled", "emu"])
    ap.add_argument("--passes", default="rec,plain")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-tail", action="store_true")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--merged", default=None)
    a = ap.parse_args()

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    tail = cfg.num_hidden_layers - 2
    if a.merged:
        load_merged(model, a.merged, tail)
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] in a.splits.split(",")]
    prompts = prompts if a.n < 0 else prompts[: a.n]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "tail": tail, "torch": torch.__version__, "merged": a.merged,
            "prompt_ids": [p["id"] for p in prompts], "stacks": {}}
    import torch._inductor.config as icfg

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for ps in a.passes.split(","):
        handles = []
        if ps == "rec":
            handles.append(model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True))
        icfg.emulate_precision_casts = a.stack == "emu"
        torch._dynamo.reset()
        fn = forward if a.stack == "eager" else torch.compile(forward, fullgraph=True)
        N, T = len(X), X[0].shape[1]
        lg_all = np.zeros((N, cfg.vocab_size), np.uint16)
        tails = np.zeros((N, T, cfg.hidden_size), np.uint16) if (ps == "rec" and a.save_tail) else None
        px, top, lse, sha = np.zeros(N), np.zeros(N, np.int64), np.zeros(N), []
        top5, top5p = np.zeros((N, 5), np.int64), np.zeros((N, 5))
        t0 = time.time()
        with torch.no_grad():
            for i, x in enumerate(X):
                REC.clear()
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
                if ps == "rec":
                    assert len(REC) == 1 and REC[0].shape[1] == T, len(REC)
                    if tails is not None:
                        tails[i] = bits(REC[0][0])
                if i % 32 == 31:
                    torch.mps.empty_cache()
        for h in handles:
            h.remove()
        info = {"s": round(time.time() - t0, 1), "N": N, "T": T, "px_mean": float(px.mean())}
        np.save(out_dir / f"{a.stack}_{ps}_logits.npy", lg_all)
        np.savez(out_dir / f"{a.stack}_{ps}.npz", px=px, top=top, lse=lse, top5=top5, top5p=top5p)
        json.dump(sha, open(out_dir / f"{a.stack}_{ps}_sha.json", "w"))
        if tails is not None:
            np.save(out_dir / f"{a.stack}_{ps}_tail.npy", tails)
        if a.gen and ps == "plain":
            # N-token greedy text, every token on THIS stack (no KV cache; for compiled stacks each new length is a
            # new shape, so dynamic=False recompiles per length: N-1 extra compiles, once per process).
            texts = []
            with torch.no_grad():
                for i, x in enumerate(X):
                    seq = torch.cat([x, torch.tensor([[int(top[i])]], device=dev)], 1)
                    for _ in range(a.gen - 1):
                        nxt = int(fn(seq).float().argmax())
                        if nxt == tok.eos_token_id or tok.convert_ids_to_tokens(nxt) == "<|im_end|>":
                            break
                        seq = torch.cat([seq, torch.tensor([[nxt]], device=dev)], 1)
                    texts.append(tok.decode(seq[0, T:], skip_special_tokens=True))
            json.dump(texts, open(out_dir / f"{a.stack}_{ps}_gen.json", "w"))
            info["gen_x_rate"] = float(np.mean([bool(XRE.match(t)) for t in texts]))
        meta["stacks"][f"{a.stack}_{ps}"] = info
        print(a.stack, ps, info, flush=True)
        torch.mps.empty_cache()
    icfg.emulate_precision_casts = False
    mp = out_dir / "meta.json"
    if mp.exists():  # one stack per process: merge the per-stack info
        old = json.load(open(mp))
        assert old["prompt_ids"] == meta["prompt_ids"] and old.get("merged") == meta["merged"]
        meta["stacks"] = {**old["stacks"], **meta["stacks"]}
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

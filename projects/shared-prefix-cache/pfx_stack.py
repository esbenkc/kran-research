"""Whole-graph runs of Qwen3-0.6B with the SHARED template-prefix cache (pfx.py) on one build per process.

Builds (as in stack-conditional/stack.py; the same weights, only the execution differs):
  eager      the GRADING build: plain PyTorch on MPS, op by op
  compiled   the SERVING build: torch.compile(fullgraph=True), default Inductor settings
  emu        UNSEEN build: compiled with torch._inductor.config.emulate_precision_casts = True
Every build reads the SAME prefix cache: K/V of `<|im_start|>user\\n` made once by the eager build (`prefix_pass`), and
runs positions 3-33 itself. Passes:
  rec    the rest graph also returns the block-26 input at positions 3-33; saved with the shared prefix rows (the
         eager prefix pass's block-26 input at 0-2) as uint16 bf16 bits [N, 34, H] = the RL training cache.
  plain  logits only (the deployed graph). Its first-token P(X) is the status metric.
Outputs as stack.py: <build>_<pass>.npz (px, top, lse, top5, top5p), _logits.npy, _sha.json, _tail.npy, meta.json.
--merged loads a trained tail first (the prefix cache is then re-made by eager with that tail).

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/pfx_stack.py --stack compiled --out raw/cache/shared --save-tail
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pfx import P, prefix_pass, rest_positions, use_pfx  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def load_merged(model, path, tail):
    sd = torch.load(path, map_location="cpu")
    for li in range(tail, model.config.num_hidden_layers):
        sub = {k[len(f"layers.{li - tail}."):]: v for k, v in sd.items() if k.startswith(f"layers.{li - tail}.")}
        missing, unexpected = model.model.layers[li].load_state_dict(sub, strict=False)
        assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)


def load_prompts(tok, splits, n, dev):
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] in splits.split(",")]
    prompts = prompts if n < 0 else prompts[:n]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    pre = X[0][:, :P]
    assert all(torch.equal(x[:, :P], pre) for x in X), "template prefix differs between prompts"
    assert tok.decode(pre[0]) == "<|im_start|>user\n"
    return prompts, X, pre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, choices=["eager", "compiled", "emu"])
    ap.add_argument("--passes", default="rec,plain")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-tail", action="store_true")
    ap.add_argument("--merged", default=None)
    ap.add_argument("--prefix-build", default="eager", choices=["eager", "compiled", "emu"],
                    help="build that makes the prefix cache: eager = the shared grading-made cache; else own cache")
    ap.add_argument("--tag", default=None, help="file prefix (default: <stack> if --prefix-build eager else <stack>_own)")
    a = ap.parse_args()
    tag = a.tag or (a.stack if a.prefix_build == "eager" else f"{a.stack}_own")

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    use_pfx(model)
    cfg = model.config
    tail = cfg.num_hidden_layers - 2
    if a.merged:
        load_merged(model, a.merged, tail)
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts, X, pre = load_prompts(tok, a.splits, a.n, dev)
    kv, pre_rows = prefix_pass(model, pre, tail, a.prefix_build)  # once per process
    pre_bits = bits(pre_rows)
    kv_sha = hashlib.sha256(b"".join(bits(k).tobytes() + bits(v).tobytes() for k, v in kv)).hexdigest()[:16]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "tail": tail, "torch": torch.__version__, "merged": a.merged, "prefix_len": P,
            "prompt_ids": [p["id"] for p in prompts], "stacks": {}, "prefix_kv_sha": {}}
    import torch._inductor.config as icfg

    def fwd_plain(x, pos, kv):
        return model(input_ids=x, position_ids=pos, use_cache=False, logits_to_keep=1, pfx_kv=kv).logits[0, -1]

    def fwd_rec(x, pos, kv):
        o = model(input_ids=x, position_ids=pos, use_cache=False, logits_to_keep=1, pfx_kv=kv,
                  output_hidden_states=True)
        return o.logits[0, -1], o.hidden_states[tail][0]

    for ps in a.passes.split(","):
        icfg.emulate_precision_casts = a.stack == "emu"
        torch._dynamo.reset()
        f = fwd_rec if ps == "rec" else fwd_plain
        fn = f if a.stack == "eager" else torch.compile(f, fullgraph=True)
        N, T = len(X), X[0].shape[1]
        pos = rest_positions(T - P, dev)
        lg_all = np.zeros((N, cfg.vocab_size), np.uint16)
        tails = np.zeros((N, T, cfg.hidden_size), np.uint16) if (ps == "rec" and a.save_tail) else None
        px, top, lse, sha = np.zeros(N), np.zeros(N, np.int64), np.zeros(N), []
        top5, top5p = np.zeros((N, 5), np.int64), np.zeros((N, 5))
        t0 = time.time()
        with torch.no_grad():
            for i, x in enumerate(X):
                r = fn(x[:, P:], pos, kv)
                lg = r[0] if ps == "rec" else r
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
                if tails is not None:
                    h = r[1]
                    assert h.shape[0] == T - P, h.shape
                    tails[i, :P] = pre_bits
                    tails[i, P:] = bits(h)
                if i % 32 == 31:
                    torch.mps.empty_cache()
        info = {"s": round(time.time() - t0, 1), "N": N, "T": T, "px_mean": float(px.mean())}
        np.save(out_dir / f"{tag}_{ps}_logits.npy", lg_all)
        np.savez(out_dir / f"{tag}_{ps}.npz", px=px, top=top, lse=lse, top5=top5, top5p=top5p)
        json.dump(sha, open(out_dir / f"{tag}_{ps}_sha.json", "w"))
        if tails is not None:
            np.save(out_dir / f"{tag}_{ps}_tail.npy", tails)
        meta["stacks"][f"{tag}_{ps}"] = info
        print(tag, ps, info, flush=True)
        torch.mps.empty_cache()
    icfg.emulate_precision_casts = False
    mp = out_dir / "meta.json"
    if mp.exists():  # one build per process: merge the per-build info
        old = json.load(open(mp))
        assert old["prompt_ids"] == meta["prompt_ids"] and old.get("merged") == meta["merged"]
        meta["stacks"] = {**old["stacks"], **meta["stacks"]}
        meta["prefix_kv_sha"] = old.get("prefix_kv_sha", {})
    meta["prefix_kv_sha"][tag] = {"prefix_build": a.prefix_build, "sha": kv_sha}
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", out_dir, "prefix_kv_sha", kv_sha)


if __name__ == "__main__":
    main()

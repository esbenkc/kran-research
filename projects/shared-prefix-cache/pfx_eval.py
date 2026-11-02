"""Whole-graph evaluation of trained tails WITH a prefix cache, one build per process, test split.

Per checkpoint (merged tail of blocks 26-27 copied IN PLACE into one model; one compiled graph per process, as in
stack-conditional/evalw.py): (1) the prefix cache is re-made by `--prefix-build` on the checkpoint's weights (eager =
the shared grading-made cache; the build itself = its own cache); (2) the build runs positions 3-33 reading it.
--prefix-build none = the stock full 34-token graph, no cache (the 'off-cache' read of the same weights).
--prefix-build eager_whole = the prefix K/V of one whole 34-token eager pass (T2: the exact grading prefix bits).

Output: data/eval/<run>/<tag>_<step>.jsonl (id, px, top1, top5, gen on --gen-steps) + .json (means, logit sha).
tag = <stack> (shared eager prefix), <stack>_own (own prefix), <stack>_nocache (stock graph).

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/pfx_eval.py --stack compiled \
        --ckpts base:0 shared_stack_s0:1024,2048 --final 12288
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pfx import P, prefix_pass, rest_positions, use_pfx  # noqa: E402
from pfx_stack import MODEL, load_prompts  # noqa: E402

XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, choices=["eager", "compiled", "emu"])
    ap.add_argument("--prefix-build", default="eager", choices=["eager", "eager_whole", "compiled", "emu", "own", "none"])
    ap.add_argument("--ckpts", nargs="+", required=True, help="run:step,step,... ; 'base:0' = untrained")
    ap.add_argument("--final", type=int, default=12288, help="the step whose weights are <run>_merged.pt")
    ap.add_argument("--runs-dir", default="raw/runs", help="dir of <run>[_at<step>]_merged.pt (relative to project)")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--gen-steps", default="")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--insert", default="", help="token ids inserted after the prefix (seam arm, D5)")
    a = ap.parse_args()
    pb = a.stack if a.prefix_build == "own" else a.prefix_build
    tag = a.stack if pb == "eager" else (f"{a.stack}_nocache" if pb == "none" else
                                         f"{a.stack}_pinwhole" if pb == "eager_whole" else f"{a.stack}_own")
    if pb == "eager" and a.stack == "eager":
        tag = "eager"
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    if pb != "none":
        use_pfx(model)
    n = model.config.num_hidden_layers
    tail = n - 2
    base_sd = {k: v.detach().clone() for k, v in model.model.layers[tail:].state_dict().items()}
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts, X, pre = load_prompts(tok, a.split, -1, dev, a.insert)
    T = X[0].shape[1]
    gen_steps = {int(s) for s in a.gen_steps.split(",") if s}
    import torch._inductor.config as icfg

    if pb == "none":
        def forward(x, kv):
            return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]
    else:
        def forward(x, kv):
            pos = torch.arange(P, x.shape[1], device=x.device)[None]
            return model(input_ids=x[:, P:], position_ids=pos, use_cache=False, logits_to_keep=1,
                         pfx_kv=kv).logits[0, -1]

    def compiled():
        icfg.emulate_precision_casts = a.stack == "emu"
        torch._dynamo.reset()
        return forward if a.stack == "eager" else torch.compile(forward, fullgraph=True)

    fn = compiled()

    def load(sd_tail: dict):
        params = dict(model.model.layers[tail:].named_parameters())
        with torch.no_grad():
            for k, p in params.items():
                src = sd_tail[k]
                assert src.shape == p.shape and src.dtype == p.dtype, k
                p.copy_(src.to(dev))

    def ckpt(run: str, step: int) -> dict:
        rd = HERE / a.runs_dir
        path = rd / (f"{run}_merged.pt" if step == a.final else f"{run}_at{step}_merged.pt")
        sd = torch.load(path, map_location="cpu")
        return {k[len("layers."):]: v for k, v in sd.items() if k.startswith("layers.") and "rotary" not in k}

    jobs = []
    for spec in a.ckpts:
        run, steps = spec.split(":")
        jobs += [(run, int(s)) for s in steps.split(",")]
    for run, step in jobs:
        od = HERE / a.out_dir / run
        od.mkdir(parents=True, exist_ok=True)
        out = od / f"{tag}_{step}.json"
        if a.skip_done and out.exists():
            print("skip", out, flush=True)
            continue
        t0 = time.time()
        load(base_sd if run == "base" else ckpt(run, step))
        kv = None
        kv_sha = None
        if pb != "none":
            kv, _ = prefix_pass(model, pre, tail, pb, full_ids=X[0])
            kv_sha = hashlib.sha256(b"".join(k.cpu().view(torch.int16).numpy().tobytes() for kk in kv for k in kk)
                                    ).hexdigest()[:16]
            if pb not in ("eager", "eager_whole"):
                fn = compiled()  # prefix_pass reset dynamo; the rest graph recompiles once per checkpoint
        rows, shas = [], []
        with torch.no_grad():
            for i, x in enumerate(X):
                lg = fn(x, kv).float()
                shas.append(hashlib.sha256(lg.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16])
                l = torch.logsumexp(lg, -1)
                px = float(torch.exp(lg[xs] - l).sum())
                t5 = torch.topk(lg, 5).indices.tolist()
                row = {"id": prompts[i]["id"], "px": px, "top1": tok.decode([t5[0]]), "top5": [tok.decode([t]) for t in t5]}
                if step in gen_steps:
                    seq = torch.cat([x, torch.tensor([[t5[0]]], device=dev)], 1)
                    for _ in range(a.gen - 1):
                        nxt = int(fn(seq, kv).float().argmax())
                        if nxt == tok.eos_token_id or tok.convert_ids_to_tokens(nxt) == "<|im_end|>":
                            break
                        seq = torch.cat([seq, torch.tensor([[nxt]], device=dev)], 1)
                    row["gen"] = tok.decode(seq[0, T:], skip_special_tokens=True)
                rows.append(row)
                if i % 32 == 31:
                    torch.mps.empty_cache()
        with open(od / f"{tag}_{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summ = {"run": run, "step": step, "stack": a.stack, "prefix_build": pb, "tag": tag, "n": len(rows),
                "px_mean": float(np.mean([r["px"] for r in rows])),
                "top1_x_rate": float(np.mean([bool(XRE.match(r["top1"])) for r in rows])),
                "prefix_kv_sha": kv_sha, "torch": torch.__version__, "s": round(time.time() - t0, 1), "logit_sha": shas}
        if step in gen_steps:
            summ["gen_x_rate"] = float(np.mean([bool(XRE.match(r["gen"])) for r in rows]))
        out.write_text(json.dumps(summ))
        print(run, step, tag, {k: summ[k] for k in ("px_mean", "top1_x_rate", "s")}, summ.get("gen_x_rate"), flush=True)


if __name__ == "__main__":
    main()

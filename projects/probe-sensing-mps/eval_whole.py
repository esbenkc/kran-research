"""Step 2 title-level metric (DEVIATIONS D3/D4): install a trained tail (blocks 26-27 + final norm, bf16) in the
WHOLE model and run it as deployed: torch.compile of the whole forward as one graph (or eager for the eager
backend), NO recorder, with the build's probe attached ('Yes' expected) and with no probe ('No' expected),
on the held-out test prompts.

    .venv/bin/python projects/probe-sensing-mps/eval_whole.py --runs compile_probe_none_s0,compile_probe_never_s0
writes raw/step2/<model>/whole/<run>.json (per-case top token, Yes/No log-probs) and prints accuracy.
A run named <backend>_<build>_<null>_s<seed> is evaluated with probe build <build> on backend <backend>
(--eval-build overrides, e.g. to score a 'never' tail under each build).
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import step1_torch as s1  # noqa: E402  (same hooks, barrier op and FP32-THROUGH forward as Step 1)

DEV = "mps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--runs", required=True, help="comma list of run names in raw/step2/<model>/runs/")
    ap.add_argument("--eval-build", default=None)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--layer", type=int, default=14)
    a = ap.parse_args()
    root = HERE / "raw" / "step2" / a.model.split("/")[-1]
    (root / "whole").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV).eval()
    L = model.config.num_hidden_layers
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    test = [p for p in prompts if p["split"] == "test"][: a.n_test]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in test]
    T = ids[0].shape[1]
    cfg = model.config
    s1.BUF["q"] = torch.empty(1, T, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16, device=DEV)
    qn = model.model.layers[a.layer].self_attn.q_norm
    attn14 = model.model.layers[a.layer].self_attn
    orig = {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.startswith(f"model.layers.{L - 2}.") or k.startswith(f"model.layers.{L - 1}.") or k == "model.norm.weight"}

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for run in a.runs.split(","):
        backend, build, null, seed = run.split("_")
        ev_build = a.eval_build or build
        sd = None if null == "base" else torch.load(root / "runs" / f"{run}.pt")  # base = untrained tail
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in orig.items():
                msd[k].copy_(v)
            if sd is not None:
                for k, v in sd.items():
                    if k == "norm.weight":
                        msd["model.norm.weight"].copy_(v)
                    else:
                        li = int(k.split(".")[1])
                        msd[f"model.layers.{L - 2 + li}." + ".".join(k.split(".")[2:])].copy_(v)
        rows = {}
        info = {}
        for cond in (0, 1):
            handles = []
            if cond == 1:
                if ev_build == "probe":
                    handles.append(qn.register_forward_hook(s1.hook_probe))
                elif ev_build == "copy":
                    handles.append(qn.register_forward_hook(s1.hook_copy))
                elif ev_build == "twok":
                    handles.append(qn.register_forward_hook(s1.hook_twok))
                elif ev_build == "fp32":
                    attn14.forward = types.MethodType(s1.fp32_through_forward, attn14)
            if backend == "compile":
                torch._dynamo.reset()
                fn = torch.compile(forward, fullgraph=True)
                with torch.no_grad():
                    fn(ids[0])  # warm-up call (compiles; COPY's buffer exists before the scored pass)
            else:
                fn = forward
            t0 = time.time()
            with torch.no_grad():
                for i, x in enumerate(ids):
                    s1.STORE.clear()
                    lg = fn(x).float()
                    lp = torch.log_softmax(lg, -1)
                    rows[(i, cond)] = {"prompt": test[i]["id"], "probe": cond, "top": int(lg.argmax()),
                                       "logp_yes": float(lp[YES]), "logp_no": float(lp[NO])}
            info[f"cond{cond}_s"] = round(time.time() - t0, 1)
            for h in handles:
                h.remove()
            if ev_build == "fp32" and cond == 1:
                del attn14.forward
            torch.mps.empty_cache()
        out_rows = []
        for i in range(len(ids)):
            for cond in (0, 1):
                r = rows[(i, cond)]
                r["correct"] = r["top"] == (YES if cond == 1 else NO)
                r["logodds_yes_no"] = r["logp_yes"] - r["logp_no"]
                out_rows.append(r)
        acc = float(np.mean([r["correct"] for r in out_rows]))
        both = float(np.mean([out_rows[k]["correct"] and out_rows[k + 1]["correct"] for k in range(0, len(out_rows), 2)]))
        sep = float(np.mean([out_rows[k + 1]["logodds_yes_no"] - out_rows[k]["logodds_yes_no"]
                             for k in range(0, len(out_rows), 2)]))
        res = {"run": run, "backend": backend, "eval_build": ev_build, "acc": acc, "both_correct": both,
               "separation_nats": sep, "n_cases": len(out_rows), "yes_id": YES, "no_id": NO, "info": info,
               "torch": torch.__version__, "rows": out_rows}
        name = run if a.eval_build is None else f"{run}__as_{ev_build}"
        (root / "whole" / f"{name}.json").write_text(json.dumps(res, indent=1))
        print("WHOLE", name, {k: res[k] for k in ("acc", "both_correct", "separation_nats")}, info, flush=True)


if __name__ == "__main__":
    main()

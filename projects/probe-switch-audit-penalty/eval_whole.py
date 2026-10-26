"""Whole-model evaluation of a merged LoRA tail, as deployed: the merged bf16 blocks 26-27 + final norm are copied
into the full Qwen3-1.7B, the whole forward is ONE torch.compile graph (fullgraph), NO recorder anywhere, and each
held-out prompt runs twice: probe attached (forward hook on block-14 self_attn.q_norm that keeps a reference) and no
probe. --emulate 1 sets torch._inductor.config.emulate_precision_casts = True (the precision-emulation build).

Per case we save: top-1 token over the full vocabulary, log-probs of 'A' and 'B', and the SHA256 of the raw bf16
answer-position logits (so bit identity between probed and unprobed can be checked without the logits).

    $PY projects/probe-switch-heads/eval_whole.py --runs lora_none_s0 --emulate 0
writes raw/step2/<model>/whole/<run>__emu<e>.json
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
import step1_torch as s1  # noqa: E402  (the same probe hook as the cache)
from tail import summarize  # noqa: E402

DEV = "mps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--runs", required=True, help="comma list of run names in raw/step2/<model>/runs/ ('base' = untrained)")
    ap.add_argument("--emulate", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--layer", type=int, default=14)
    a = ap.parse_args()
    import torch._inductor.config as icfg
    icfg.emulate_precision_casts = bool(a.emulate)
    root = HERE / "raw" / "step2" / a.model.split("/")[-1]
    (root / "whole").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV).eval()
    L = model.config.num_hidden_layers
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    test = [p for p in prompts if p["split"] == a.split]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in test]
    qn = model.model.layers[a.layer].self_attn.q_norm
    orig = {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.startswith(f"model.layers.{L - 2}.") or k.startswith(f"model.layers.{L - 1}.") or k == "model.norm.weight"}

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for run in a.runs.split(","):
        sd = None if run == "base" else torch.load(root / "runs" / f"{run}_merged.pt")
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in orig.items():
                msd[k].copy_(v)
            if sd is not None:
                n_set = 0
                for k, v in sd.items():
                    if k == "norm.weight":
                        msd["model.norm.weight"].copy_(v); n_set += 1
                    elif k.startswith("layers."):
                        li = int(k.split(".")[1])
                        msd[f"model.layers.{L - 2 + li}." + ".".join(k.split(".")[2:])].copy_(v); n_set += 1
                assert n_set == len(orig), (n_set, len(orig))
        rows, info = {}, {}
        for cond in (0, 1):
            handle = qn.register_forward_hook(s1.hook_probe) if cond == 1 else None
            torch._dynamo.reset()
            fn = torch.compile(forward, fullgraph=True)
            with torch.no_grad():
                fn(ids[0])  # warm-up (compiles)
            t0 = time.time()
            stored_ok = True
            rep_sha = []  # second warm pass: determinism floor (must equal the first pass bit for bit)
            with torch.no_grad():
                for rep in range(2):
                    for i, x in enumerate(ids):
                        s1.STORE.clear()
                        lg16 = fn(x)
                        sha = hashlib.sha256(lg16.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
                        if cond == 1:
                            stored_ok &= len(s1.STORE) == 1
                        if rep == 1:
                            rep_sha.append(sha == rows[(i, cond)]["logits_sha256"])
                            continue
                        lg = lg16.float()
                        lp = torch.log_softmax(lg, -1)
                        rows[(i, cond)] = {"prompt": test[i]["id"], "split": a.split, "probe": cond,
                                           "top": int(lg.argmax()), "a_id": A_ID, "b_id": B_ID,
                                           "logp_a": float(lp[A_ID]), "logp_b": float(lp[B_ID]),
                                           "lo": float(lp[A_ID] - lp[B_ID]), "logits_sha256": sha}
                        if i % 16 == 15:
                            torch.mps.empty_cache()
            info[f"cond{cond}_s"] = round(time.time() - t0, 1)
            info[f"cond{cond}_repeat_changed_prompts"] = int(len(rep_sha) - sum(rep_sha))
            if cond == 1:
                info["probe_stored_ok"] = bool(stored_ok)
            if handle is not None:
                handle.remove()
            torch.mps.empty_cache()
        out_rows = [rows[(i, c)] for i in range(len(ids)) for c in (0, 1)]
        res = summarize(out_rows)
        res["bit_identical_prompts"] = int(sum(out_rows[k]["logits_sha256"] == out_rows[k + 1]["logits_sha256"]
                                              for k in range(0, len(out_rows), 2)))
        out = {"run": run, "emulate": a.emulate, "split": a.split, **res, "info": info, "torch": torch.__version__,
               "rows": out_rows}
        (root / "whole" / f"{run}__emu{a.emulate}.json").write_text(json.dumps(out, indent=1))
        print("WHOLE", run, f"emu{a.emulate}", res, info, flush=True)


if __name__ == "__main__":
    main()

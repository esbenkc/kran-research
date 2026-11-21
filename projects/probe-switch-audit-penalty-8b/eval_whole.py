"""Whole-model evaluation of a trained LoRA tail, as deployed (the status number of the switch clause). Port of the 1.7B
parent's eval_whole.py (probe-switch-heads, unchanged in oracle-1026-audit-penalty): the adapter is merged into bf16
blocks FIRST, FIRST+1 (8B: 34-35) and copied into the full model; the whole forward is ONE torch.compile graph
(fullgraph), NO recorder anywhere, and each held-out prompt runs twice: probe attached (forward hook on block
PROBE_LAYER self_attn.q_norm that keeps a reference; the cache's probe) and no probe.

The compiled function returns the answer-position logits the way Qwen3ForCausalLM computes them with
logits_to_keep=1 (lm_head on the last hidden state, bf16) AND that last final-norm hidden state; the fp32 row is
hidden.float() @ lm_head.float().T (same merged weights). Per case: top-1 over the full vocabulary, log-probs of 'A'
and 'B' (bf16 and fp32), SHA256 of the raw bf16 logits. A second warm pass checks run-to-run determinism.

    AP_MODEL=Qwen/Qwen3-8B $PY eval_whole.py --runs warm_s0=adapters_in/switch_s0_adapter.pt,ao_s0=data/adapters/ao_s0_adapter.pt
writes OUT/whole/<run>.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import step1_torch as s1  # noqa: E402  (the same probe hook as the cache)
from audit import merged_tail  # noqa: E402
from cfgm import FIRST, MODEL, OUT, PROBE_LAYER  # noqa: E402
from tail import summarize  # noqa: E402

DEV = "mps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of name=adapter.pt")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    (OUT / "whole").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    test = [p for p in prompts if p["split"] == a.split]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in test]
    qn = model.model.layers[PROBE_LAYER].self_attn.q_norm
    keys = [k for k in model.state_dict() if k.startswith(f"model.layers.{FIRST}.")
            or k.startswith(f"model.layers.{FIRST + 1}.") or k == "model.norm.weight"]
    w32 = model.lm_head.weight.float()

    def forward(x):
        h = model.model(input_ids=x, use_cache=False).last_hidden_state[:, -1:, :]
        return model.lm_head(h)[0, -1], h[0, -1]

    for spec in a.runs.split(","):
        run, adapter = spec.split("=")
        tail, _ = merged_tail(MODEL, HERE / adapter)
        sd = {k: v for k, v in tail.state_dict().items() if "rotary" not in k}
        with torch.no_grad():
            msd = model.state_dict()
            n_set = 0
            for k, v in sd.items():
                if k == "norm.weight":
                    msd["model.norm.weight"].copy_(v); n_set += 1
                elif k.startswith("layers."):
                    li = int(k.split(".")[1])
                    msd[f"model.layers.{FIRST + li}." + ".".join(k.split(".")[2:])].copy_(v); n_set += 1
            assert n_set == len(keys), (n_set, len(keys))
        del tail, sd
        torch.mps.empty_cache()
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
                        lg16, hl = fn(x)
                        sha = hashlib.sha256(lg16.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
                        if cond == 1:
                            stored_ok &= len(s1.STORE) == 1
                        if rep == 1:
                            rep_sha.append(sha == rows[(i, cond)]["logits_sha256"])
                            continue
                        lg = lg16.float()
                        lp = torch.log_softmax(lg, -1)
                        lg32 = hl.float() @ w32.T
                        lp32 = torch.log_softmax(lg32, -1)
                        rows[(i, cond)] = {"prompt": test[i]["id"], "split": a.split, "probe": cond,
                                           "top": int(lg.argmax()), "a_id": A_ID, "b_id": B_ID,
                                           "logp_a": float(lp[A_ID]), "logp_b": float(lp[B_ID]),
                                           "lo": float(lp[A_ID] - lp[B_ID]), "top32": int(lg32.argmax()),
                                           "lo32": float(lp32[A_ID] - lp32[B_ID]), "logits_sha256": sha}
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
        out = {"run": run, "adapter": adapter, "adapter_sha256": hashlib.sha256((HERE / adapter).read_bytes()).hexdigest(),
               "split": a.split, **res, "info": info, "torch": torch.__version__, "model": MODEL, "rows": out_rows}
        (OUT / "whole" / f"{run}.json").write_text(json.dumps(out, indent=1))
        print("WHOLE", run, json.dumps({k: res[k] for k in ("switch_strict", "switch_correct", "switch_strict32",
                                                             "share_unprobed_B", "shift_mean")}), info, flush=True)


if __name__ == "__main__":
    main()

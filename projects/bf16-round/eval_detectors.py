"""Score the 3 saved probe detectors from probe-sensing-mps (trained tails: blocks 26-27 + final norm) under
each FIX. The tail is installed in the whole Qwen3-1.7B, the whole forward is compiled as one graph (no
recorder), and each of 128 held-out test prompts runs without the probe ('No' is correct) and with the probe
('Yes' is correct). The FIX is in q_norm in both conditions (bits.py make_forward).

    .venv/bin/python projects/bf16-round/eval_detectors.py --emulate 0 --fixes none,round16 \
        --runs compile_probe_none_s0,compile_probe_none_s1,compile_probe_none_s2,compile_probe_never_s0

Detectors: bf16env/detectors/<run>.pt (cp -c of probe-sensing-mps raw/step2/Qwen3-1.7B/runs/<run>.pt).
Writes raw/detectors/<model>/emu<e>/<fix>__<run>.json (per case: top token, log-probs of Yes/No, SHA256 of the
all-position logit bits) and <fix>__<run>_last.npy (uint16 [prompt, probe 0/1, vocab]: answer-position logit bits).
"""
import argparse
import gc
import hashlib
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
import bits as B  # noqa: E402  (same fixes, probe hook and barrier op)

DEV = "mps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--emulate", type=int, choices=[0, 1], required=True)
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--det-dir", default=str(HERE.parents[2] / "bf16env" / "detectors"))
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--layer", type=int, default=14)
    a = ap.parse_args()

    import torch._inductor.config as icfg
    icfg.emulate_precision_casts = bool(a.emulate)
    icfg.fx_graph_cache = False

    out_dir = HERE / "raw" / "detectors" / a.model.split("/")[-1] / f"emu{a.emulate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV).eval()
    L = model.config.num_hidden_layers
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    test = [p for p in prompts if p["split"] == "test"][: a.n_test]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in test]
    qn = model.model.layers[a.layer].self_attn.q_norm
    orig = {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.startswith(f"model.layers.{L - 2}.") or k.startswith(f"model.layers.{L - 1}.") or k == "model.norm.weight"}

    def forward(x):
        out = model.model(input_ids=x, use_cache=False)
        return model.lm_head(out.last_hidden_state[0])

    for run in a.runs.split(","):
        sd = torch.load(Path(a.det_dir) / f"{run}.pt")
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in orig.items():
                msd[k].copy_(v)
            for k, v in sd.items():
                if k == "norm.weight":
                    msd["model.norm.weight"].copy_(v)
                else:
                    li = int(k.split(".")[1])
                    msd[f"model.layers.{L - 2 + li}." + ".".join(k.split(".")[2:])].copy_(v)
        del sd
        for fix in a.fixes.split(","):
            qn.forward = types.MethodType(B.make_forward(fix), qn)
            rows, info = {}, {}
            last = np.zeros((len(ids), 2, model.config.vocab_size), np.uint16)
            for cond in (0, 1):
                h = qn.register_forward_hook(B.hook_probe) if cond else None
                torch._dynamo.reset()
                fn = torch.compile(forward, fullgraph=True)
                t0 = time.time()
                with torch.no_grad():
                    fn(ids[0])  # warm-up (compiles)
                    for i, x in enumerate(ids):
                        B.STORE.clear()
                        lg = fn(x)
                        lb = B.bits(lg)
                        last[i, cond] = lb[-1]
                        lp = torch.log_softmax(lg[-1].float(), -1)
                        rows[(i, cond)] = {"prompt": test[i]["id"], "probe": cond, "top": int(lg[-1].argmax()),
                                           "logp_yes": float(lp[YES]), "logp_no": float(lp[NO]),
                                           "sha256_all_position_logits": hashlib.sha256(lb.tobytes()).hexdigest()}
                        if i % 16 == 15:
                            torch.mps.empty_cache()
                info[f"cond{cond}_s"] = round(time.time() - t0, 1)
                if h is not None:
                    h.remove()
                torch.mps.empty_cache()
            del qn.forward
            out_rows = []
            for i in range(len(ids)):
                for cond in (0, 1):
                    r = rows[(i, cond)]
                    r["correct"] = r["top"] == (YES if cond == 1 else NO)
                    r["logodds_yes_no"] = r["logp_yes"] - r["logp_no"]
                    out_rows.append(r)
            acc = float(np.mean([r["correct"] for r in out_rows]))
            pairs_same = int(sum(out_rows[k]["sha256_all_position_logits"] == out_rows[k + 1]["sha256_all_position_logits"]
                                 for k in range(0, len(out_rows), 2)))
            sep = float(np.mean([out_rows[k + 1]["logodds_yes_no"] - out_rows[k]["logodds_yes_no"]
                                 for k in range(0, len(out_rows), 2)]))
            res = {"run": run, "fix": fix, "emulate_precision_casts": bool(a.emulate), "acc": acc,
                   "prompts_bit_identical_probe_vs_none": pairs_same, "n_prompts": len(ids),
                   "separation_nats": sep, "yes_id": YES, "no_id": NO, "info": info, "torch": torch.__version__,
                   "rows": out_rows}
            (out_dir / f"{fix}__{run}.json").write_text(json.dumps(res, indent=1))
            np.save(out_dir / f"{fix}__{run}_last.npy", last)
            del last, rows, out_rows
            gc.collect()
            torch._dynamo.reset()
            torch.mps.empty_cache()
            print("DET", fix, run, {k: res[k] for k in ("acc", "prompts_bit_identical_probe_vs_none", "separation_nats")},
                  info, flush=True)
            del res


if __name__ == "__main__":
    main()

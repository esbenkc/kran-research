"""Title score (PREREG): the LoRA merged into bf16 weights inside the full Qwen3-1.7B, the whole forward as ONE
torch.compile graph (fullgraph), NO recorder anywhere, each held-out prompt run with the probe attached (forward hook
on block-14 self_attn.q_norm that keeps a reference) and without it. A second warm pass checks determinism.
(from probe-switch-heads/eval_whole.py; merge = train.py LoRALinear.merged, the same bf16 rounding.)

    $PY eval_whole.py --runs all8_none_s0,mlp8_none_s0      ('base' = untrained model)
writes data/whole/<run>.json (per-prompt rows: top-1/top-2 ids and logits, logit/logp of A and B, logits sha256)
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
from tail import make_row, summarize  # noqa: E402

DEV = "mps"
STORE: list = []


def hook_probe(mod, inp, out):
    STORE.append(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--mark", action="store_true", help="MARK-8: condition 1 = visible marker token, no probe hook")
    a = ap.parse_args()
    (HERE / "data" / "whole").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV).eval()
    L = model.config.num_hidden_layers
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    test = [p for p in prompts if p["split"] == a.split]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in test]
    qn = model.model.layers[a.layer].self_attn.q_norm
    msd = model.state_dict()
    orig = {k: v.detach().clone() for k, v in msd.items() if any(k.startswith(f"model.layers.{li}.") for li in range(20, L))}
    from cache import MARK_POS, MARK_WORD  # noqa: E402  (the same marker as cache.py --pass M)
    mark_id = tok.encode(MARK_WORD, add_special_tokens=False)[0]

    def mark(x):
        y = x.clone()
        y[0, MARK_POS] = mark_id
        return y

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for run in a.runs.split(","):
        with torch.no_grad():
            for k, v in orig.items():
                msd[k].copy_(v)
            n_set = 0
            if run != "base":
                meta = json.load(open(HERE / "data" / "runs" / f"{run}.json"))
                first = 27 if meta["args"]["arm"] == "mlp27" else 20
                scale = meta["args"]["alpha"] / meta["args"]["rank"]
                ad = torch.load(HERE / "raw" / "runs" / f"{run}_adapter.pt")
                for k in sorted(x[:-2] for x in ad if x.endswith(".A")):  # layers.<i>.<mlp|self_attn>.<proj>
                    li = int(k.split(".")[1])
                    wk = f"model.layers.{first + li}." + ".".join(k.split(".")[2:]) + ".weight"
                    A, B = ad[k + ".A"].to(DEV), ad[k + ".B"].to(DEV)
                    msd[wk].copy_((orig[wk].float() + scale * (B @ A)).to(torch.bfloat16))
                    n_set += 1
                assert n_set == len(meta["lora_modules"]), (n_set, len(meta["lora_modules"]))
        rows, info = {}, {"merged_modules": n_set}
        for cond in (0, 1):
            handle = qn.register_forward_hook(hook_probe) if (cond == 1 and not a.mark) else None
            torch._dynamo.reset()
            fn = torch.compile(forward, fullgraph=True)
            with torch.no_grad():
                fn(ids[0])
            t0 = time.time()
            ok, same = True, []
            with torch.no_grad():
                for rep in range(2):
                    for i, x in enumerate(ids):
                        STORE.clear()
                        lg16 = fn(mark(x) if (cond == 1 and a.mark) else x)
                        sha = hashlib.sha256(lg16.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
                        if cond == 1:
                            ok &= len(STORE) == (0 if a.mark else 1)
                        if rep == 1:
                            same.append(sha == rows[(i, cond)]["logits_sha256"])
                            continue
                        r = make_row(lg16.float().cpu(), test[i]["id"], cond, A_ID, B_ID)
                        r["logits_sha256"] = sha
                        rows[(i, cond)] = r
                        if i % 16 == 15:
                            torch.mps.empty_cache()
            info[f"cond{cond}_s"] = round(time.time() - t0, 1)
            info[f"cond{cond}_repeat_changed_prompts"] = int(len(same) - sum(same))
            if cond == 1:
                info["probe_stored_ok"] = bool(ok)
            if handle is not None:
                handle.remove()
            torch.mps.empty_cache()
        out_rows = [rows[(i, c)] for i in range(len(ids)) for c in (0, 1)]
        res = summarize(out_rows)
        res["bit_identical_prompts"] = int(sum(out_rows[k]["logits_sha256"] == out_rows[k + 1]["logits_sha256"]
                                              for k in range(0, len(out_rows), 2)))
        out = {"run": run, "split": a.split, **res, "info": info, "torch": torch.__version__, "rows": out_rows}
        (HERE / "data" / "whole" / f"{run}.json").write_text(json.dumps(out))
        print("WHOLE", run, res, info, flush=True)


if __name__ == "__main__":
    main()

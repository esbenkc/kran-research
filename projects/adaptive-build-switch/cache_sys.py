"""Block-26 input caches with a system turn (penalty (c) training texts), one build per process.

Same whole-graph forward as stack-conditional/stack.py (bf16, batch 1, MPS; eager = op by op, compiled =
torch.compile(fullgraph=True), default Inductor), with prefix = sysprompts.json train[i].prefix placed before the
trained user prompt. A pre-hook on block 26 keeps its input (uint16 bf16 bits). Every text has its own length, so
the compiled build compiles once per text (dynamo reset between texts). Rows = train + dev prompts, base meta order.

    PY cache_sys.py --stack eager     -> raw/cache/sys/eager_t<i>_tail.npy  [N, T_i, 1024], meta.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
REC: list = []


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t):
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, choices=["eager", "compiled"])
    ap.add_argument("--role", default="train")
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--out", default="raw/cache/sys")
    ap.add_argument("--texts", default="", help="comma list of text indices (default all)")
    a = ap.parse_args()
    dev = "mps"
    sp = json.load(open(HERE / "data" / "sysprompts.json"))
    rows = sp[a.role]
    tsel = [int(i) for i in a.texts.split(",")] if a.texts else list(range(len(rows)))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    tail = model.config.num_hidden_layers - 2
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    base_ids = json.load(open(HERE / "raw" / "cache" / "base" / "meta.json"))["prompt_ids"]
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")}
    ids = [p for p in base_ids if prompts[p]["split"] in a.splits.split(",")]
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True)
    torch._dynamo.config.cache_size_limit = 64

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    info = {}
    for ti in tsel:
        pre = rows[ti]["prefix"]
        X = []
        for p in ids:
            full = tok(pre + prompts[p]["text"], add_special_tokens=False)["input_ids"]
            assert full[-34:] == tok(prompts[p]["text"], add_special_tokens=False)["input_ids"], p
            X.append(torch.tensor([full], device=dev))
        T = X[0].shape[1]
        assert all(x.shape[1] == T for x in X)
        torch._dynamo.reset()
        fn = forward if a.stack == "eager" else torch.compile(forward, fullgraph=True)
        tails = np.zeros((len(X), T, model.config.hidden_size), np.uint16)
        px = np.zeros(len(X))
        t0 = time.time()
        with torch.no_grad():
            for i, x in enumerate(X):
                REC.clear()
                lg = fn(x).float()
                assert len(REC) == 1 and REC[0].shape[1] == T
                tails[i] = bits(REC[0][0])
                px[i] = float(torch.softmax(lg, -1)[xs].sum())
                if i % 32 == 31:
                    torch.mps.empty_cache()
        np.save(out / f"{a.stack}_{a.role}_t{ti}_tail.npy", tails)
        np.save(out / f"{a.stack}_{a.role}_t{ti}_px.npy", px)
        info[str(ti)] = {"T": T, "s": round(time.time() - t0, 1), "px_mean": float(px.mean()),
                         "sha": hashlib.sha256(tails.tobytes()).hexdigest()[:16]}
        print(a.stack, ti, info[str(ti)], flush=True)
    mp = out / "meta.json"
    meta = json.load(open(mp)) if mp.exists() else {"prompt_ids": ids, "role": a.role,
                                                     "texts": [r["text"] for r in rows], "stacks": {}}
    assert meta["prompt_ids"] == ids and meta["role"] == a.role
    meta["stacks"].setdefault(a.stack, {}).update(info)
    json.dump(meta, open(mp, "w"), indent=1)


if __name__ == "__main__":
    main()

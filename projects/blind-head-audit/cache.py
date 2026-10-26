"""Neutral-prompt cache: block-26 inputs of the WHOLE compiled model with and without the probe attached.

Same graph as the training caches of all three source projects (probe-switch-heads/step1_torch.py,
numeric-chameleon-rl/cache.py): torch.compile(fullgraph=True), batch 1, bf16, one token shape, logits_to_keep=1, a
recorder pre-hook on block n_layers-2 in EVERY build (the recorder is itself a reader, so it is in both conditions),
and the probe = a forward hook on model.layers[14].self_attn.q_norm that keeps a reference to its output (read-only).

Builds:
  none        no probe
  probe       LEAKY probe (stock Inductor)
  none_emu    no probe,   torch._inductor.config.emulate_precision_casts = True
  probe_emu   leaky hook, emulate_precision_casts = True (the precision-emulation build: expected bit-identical to none_emu)

Blocks 0-25 are the untouched base model in every trained model of this audit (all training was on blocks 26-27),
so this cache serves every model of one base size.

    $PY cache.py --model Qwen/Qwen3-1.7B
writes raw/neutral/<model>/<build>_tail.npy (uint16 bf16 bits [N, T, H]), <build>_logits.npy, meta.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch._inductor.config as icfg
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
STORE: list = []
REC: list = []


def hook_probe(mod, inp, out):
    STORE.append(out)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t):
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--builds", default="none,probe,none_emu,probe_emu")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--prompts", default=str(HERE / "data" / "neutral.jsonl"))
    a = ap.parse_args()
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    tail = model.config.num_hidden_layers - 2
    prompts = [json.loads(l) for l in open(a.prompts)]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = ids[0].shape[1]
    assert all(x.shape[1] == T for x in ids)
    qn = model.model.layers[a.layer].self_attn.q_norm
    out_dir = HERE / "raw" / "neutral" / a.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "layer": a.layer, "tail": tail, "T": T, "n": len(ids), "torch": torch.__version__,
            "prompt_ids": [p["id"] for p in prompts], "builds": {}}

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for build in a.builds.split(","):
        handles = [model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True)]
        if build.startswith("probe"):
            handles.append(qn.register_forward_hook(hook_probe))
        icfg.emulate_precision_casts = build.endswith("_emu")
        torch._dynamo.reset()
        fn = torch.compile(forward, fullgraph=True)
        tails = np.zeros((len(ids), T, model.config.hidden_size), np.uint16)
        logits = np.zeros((len(ids), model.config.vocab_size), np.uint16)
        ok = True
        t0 = time.time()
        with torch.no_grad():
            for i, x in enumerate(ids):
                STORE.clear(); REC.clear()
                lg = fn(x)
                logits[i] = bits(lg)
                assert len(REC) == 1
                tails[i] = bits(REC[0][0])
                ok &= len(STORE) == (1 if build.startswith("probe") else 0)
                if i % 16 == 15:
                    torch.mps.empty_cache()
        for h in handles:
            h.remove()
        np.save(out_dir / f"{build}_tail.npy", tails)
        np.save(out_dir / f"{build}_logits.npy", logits)
        info = {"s": round(time.time() - t0, 1), "hook_ran_ok": bool(ok),
                "tail_sha256": hashlib.sha256(tails.tobytes()).hexdigest(),
                "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest()}
        meta["builds"][build] = info
        print(build, info, flush=True)
        torch.mps.empty_cache()
    icfg.emulate_precision_casts = False
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

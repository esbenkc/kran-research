"""T4 step 1 (only if the title arm leaves >= 1 changed prompt): cache the block-26 inputs of all 576 prompts for one
FIX, unprobed and probed, compiled as one graph, with the block-26 input recorder in BOTH conditions (as in
probe-sensing-mps D2), so a fresh detector can be trained on this build with the probe-sensing-mps recipe.

    .venv/bin/python projects/bf16-round/cache_tail.py --fix round16
writes raw/step2/Qwen3-1.7B/<fix>_C/{none,probe}_rec_tail.npy (uint16 [576, 71, 2048]) + meta.json, the layout that
tail_1022.py (= probe-sensing-mps train_tail.py) reads with --backend <fix> --build probe.
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
import bits as B  # noqa: E402

REC: list = []


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--fix", required=True)
    ap.add_argument("--layer", type=int, default=14)
    a = ap.parse_args()
    import torch._inductor.config as icfg
    icfg.fx_graph_cache = False
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    L = model.config.num_hidden_layers
    tail = L - 2
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = ids[0].shape[1]
    qn = model.model.layers[a.layer].self_attn.q_norm
    qn.forward = types.MethodType(B.make_forward(a.fix), qn)
    out_dir = HERE / "raw" / "step2" / a.model.split("/")[-1] / f"{a.fix}_C"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "fix": a.fix, "layer": a.layer, "tail": tail, "T": T,
            "prompt_ids": [p["id"] for p in prompts], "torch": torch.__version__, "builds": {}}

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for build in ("none", "probe"):
        hs = [model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True)]
        if build == "probe":
            hs.append(qn.register_forward_hook(B.hook_probe))
        torch._dynamo.reset()
        fn = torch.compile(forward, fullgraph=True)
        arr = np.lib.format.open_memmap(out_dir / f"{build}_rec_tail.npy", mode="w+", dtype=np.uint16,
                                        shape=(len(ids), T, model.config.hidden_size))
        t0 = time.time()
        with torch.no_grad():
            REC.clear(); B.STORE.clear()
            fn(ids[0])  # compile
            for i, x in enumerate(ids):
                REC.clear(); B.STORE.clear()
                fn(x)
                assert len(REC) == 1
                arr[i] = B.bits(REC[0][0])
                if i % 16 == 15:
                    arr.flush()
                    torch.mps.empty_cache()
        arr.flush()
        del arr
        for h in hs:
            h.remove()
        meta["builds"][f"{build}_rec"] = {"total_s": round(time.time() - t0, 1)}
        print(build, meta["builds"][f"{build}_rec"], flush=True)
    a0 = np.load(out_dir / "none_rec_tail.npy", mmap_mode="r")
    a1 = np.load(out_dir / "probe_rec_tail.npy", mmap_mode="r")
    meta["prompts_changed_at_block26_input"] = int(sum(not np.array_equal(a0[i], a1[i]) for i in range(len(ids))))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print("changed at block-26 input:", meta["prompts_changed_at_block26_input"], "/", len(ids))


if __name__ == "__main__":
    main()

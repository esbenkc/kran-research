"""Cache the leaky compiled build (torch.compile, Inductor, MPS, bf16) of PS_MODEL with and without the read-only probe.

Port of the parent probe-switch-frozen-heads/cache.py (Qwen3-1.7B) to Qwen3-8B; only the constants come from cfgm.py.
Probe = forward hook on block-PROBE_LAYER self_attn.q_norm that keeps a reference to its output (read-only); this is
the 'leaky build' of probe-switch-heads / probe-switch-audit-penalty / probe-switch-frozen-heads (same prompts, splits).
One pass per condition (none, probe), each its own torch.compile(fullgraph=True) graph, every prompt once:
  A  ('train cache')  one recorder (pre-hook) at the block-FIRST input -> [704, 80, HIDDEN] bf16 bits + answer logits.
     The 8-block arms train on exactly these inputs (eager tail, tail.py).
  M  (MARK-8 capacity control; only if MLP-8 s0 is null) like A, but NO probe in either condition: 'none' = the
     prompt, 'probe' = the same prompt with the body token at position MARK_POS replaced by MARK_WORD (a visible
     trigger, same 80 tokens). Files keep the none/probe names so train.py --input M reads them as is.
The parent's pass B (recorders at 29 sites, a different compiled graph; PREREG v1.3 demoted it to a labelled column)
is not built at 8B.

    $PY cache.py --pass A     (and --pass M)
writes CACHE/{none,probe}_{A,M}_tailin.npy, {none,probe}_{A,M}_logits.npy, meta_{A,M}.json
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfgm import CACHE, FIRST, MODEL, PROBE_LAYER, PROMPTS

STORE: list = []
REC: list = []
MARK_POS, MARK_WORD = 40, " banana"  # position 40 lies inside the 40-token body (positions 3..42)


def hook_probe(mod, inp, out):
    STORE.append(out)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pass", dest="ps", choices=["A", "M"], required=True)
    ap.add_argument("--layer", type=int, default=PROBE_LAYER)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out", default=str(CACHE))
    a = ap.parse_args()

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    prompts = [json.loads(l) for l in open(PROMPTS)]
    prompts = prompts if a.n < 0 else prompts[: a.n]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = ids[0].shape[1]
    assert all(x.shape[1] == T for x in ids) and T == 80
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert all(int(x[0, 71]) == im_end for x in ids), "position 71 must be <|im_end|>"
    mark_id = tok.encode(MARK_WORD, add_special_tokens=False)
    assert len(mark_id) == 1, mark_id
    mark_id = mark_id[0]
    from pathlib import Path
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    qn = model.model.layers[a.layer].self_attn.q_norm
    site = model.model.layers[FIRST]
    meta = {"model": a.model, "pass": a.ps, "probe_layer": a.layer, "first": FIRST, "T": T,
            "site": f"in{FIRST}", "prompt_ids": [p["id"] for p in prompts], "torch": torch.__version__, "builds": {}}
    if a.ps == "M":
        meta.update({"mark_word": MARK_WORD, "mark_id": mark_id, "mark_pos": MARK_POS,
                     "replaced_tokens": [tok.decode([int(x[0, MARK_POS])]) for x in ids[:8]]})

    def marked(x):
        y = x.clone()
        y[0, MARK_POS] = mark_id
        return y

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for build in ("none", "probe"):
        handles = [site.register_forward_pre_hook(pre_rec, with_kwargs=True)]
        if build == "probe" and a.ps == "A":
            handles.append(qn.register_forward_hook(hook_probe))
        torch._dynamo.reset()
        fn = torch.compile(forward, fullgraph=True)
        logits = np.zeros((len(ids), cfg.vocab_size), np.uint16)
        tmp = out_dir / f"{build}_{a.ps}_tailin.npy.part"
        store = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(len(ids), T, cfg.hidden_size))
        ok = True
        t0 = time.time()
        info = {}
        with torch.no_grad():
            for i, x in enumerate(ids):
                STORE.clear(); REC.clear()
                lg = fn(marked(x) if (a.ps == "M" and build == "probe") else x)
                logits[i] = bits(lg)
                assert len(REC) == 1, len(REC)
                if build == "probe" and a.ps == "A":
                    ok &= len(STORE) == 1
                store[i] = bits(REC[0][0])
                if i == 0:
                    info["first_call_s"] = round(time.time() - t0, 1)
                if i % 16 == 15:
                    torch.mps.empty_cache()
        info["total_s"] = round(time.time() - t0, 1)
        info["probe_stored_ok"] = bool(ok) if (build == "probe" and a.ps == "A") else None
        for h in handles:
            h.remove()
        store.flush(); del store
        tmp.replace(out_dir / f"{build}_{a.ps}_tailin.npy")
        np.save(out_dir / f"{build}_{a.ps}_logits.npy", logits)
        meta["builds"][build] = info
        print(build, a.ps, info, flush=True)
        torch.mps.empty_cache()
    json.dump(meta, open(out_dir / f"meta_{a.ps}.json", "w"), indent=1)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()

"""Cache the leaky compiled build (torch.compile, Inductor, MPS, bf16) of Qwen3-1.7B with and without the read-only probe.

Probe = forward hook on block-14 self_attn.q_norm that keeps a reference to its output (read-only); this is the
'leaky build' of probe-switch-heads / probe-switch-audit-penalty (same prompts, same splits).
Two passes per condition (none, probe), each its own torch.compile(fullgraph=True) graph, every prompt once:
  A  ('train cache')  one recorder (pre-hook) at the block-20 input -> [704, 80, 2048] bf16 bits, + answer logits.
     The 8-block arms train on exactly these inputs (eager tail, tail.py).
  B  ('whole-graph map') recorders at every block input 14..27, every mid-block residual 14..27 (input of
     post_attention_layernorm = residual after the attention add) and the final-norm input; positions 71
     (<|im_end|>) and 79 (answer) only -> [704, 2, 29, 2048]. Recorders change the compiled graph, so B is the
     'whole graph with recorders' column of the map; the eager-tail column comes from map_states.py.

    $PY cache.py --pass A     (and --pass B)
writes raw/cache/{none,probe}_A_b20.npy, {none,probe}_A_logits.npy, {none,probe}_B_states.npy, meta_{A,B}.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
STORE: list = []
REC: list = []
POS = (71, 79)
FIRST = 20


def hook_probe(mod, inp, out):
    STORE.append(out)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pass", dest="ps", choices=["A", "B"], required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out", default=str(HERE / "raw" / "cache"))
    a = ap.parse_args()

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = prompts if a.n < 0 else prompts[: a.n]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = ids[0].shape[1]
    assert all(x.shape[1] == T for x in ids) and T == 80
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert all(int(x[0, 71]) == im_end for x in ids), "position 71 must be <|im_end|>"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    qn = model.model.layers[a.layer].self_attn.q_norm
    # recorder sites, in forward order
    if a.ps == "A":
        sites = [("in", FIRST, model.model.layers[FIRST])]
    else:
        sites = []
        for li in range(a.layer, L):
            sites.append(("in", li, model.model.layers[li]))
            sites.append(("mid", li, model.model.layers[li].post_attention_layernorm))
        sites.append(("final", L, model.model.norm))
    meta = {"model": a.model, "pass": a.ps, "probe_layer": a.layer, "T": T, "positions": list(POS),
            "sites": [f"{k}{li}" for k, li, _ in sites], "prompt_ids": [p["id"] for p in prompts],
            "torch": torch.__version__, "builds": {}}

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for build in ("none", "probe"):
        handles = [m.register_forward_pre_hook(pre_rec, with_kwargs=True) for _, _, m in sites]
        if build == "probe":
            handles.append(qn.register_forward_hook(hook_probe))
        torch._dynamo.reset()
        fn = torch.compile(forward, fullgraph=True)
        logits = np.zeros((len(ids), cfg.vocab_size), np.uint16)
        if a.ps == "A":
            store = np.lib.format.open_memmap(out_dir / f"{build}_A_b20.npy", mode="w+", dtype=np.uint16,
                                              shape=(len(ids), T, cfg.hidden_size))
        else:
            store = np.zeros((len(ids), len(POS), len(sites), cfg.hidden_size), np.uint16)
        ok = True
        t0 = time.time()
        info = {}
        with torch.no_grad():
            for i, x in enumerate(ids):
                STORE.clear(); REC.clear()
                lg = fn(x)
                logits[i] = bits(lg)
                assert len(REC) == len(sites), (len(REC), len(sites))
                if build == "probe":
                    ok &= len(STORE) == 1
                if a.ps == "A":
                    store[i] = bits(REC[0][0])
                else:
                    st = torch.stack([r[0, list(POS)] for r in REC], 1)  # [pos, site, H]
                    store[i] = bits(st)
                if i == 0:
                    info["first_call_s"] = round(time.time() - t0, 1)
                if i % 16 == 15:
                    torch.mps.empty_cache()
        info["total_s"] = round(time.time() - t0, 1)
        info["probe_stored_ok"] = bool(ok) if build == "probe" else None
        for h in handles:
            h.remove()
        if a.ps == "A":
            store.flush(); del store
            np.save(out_dir / f"{build}_A_logits.npy", logits)
        else:
            np.save(out_dir / f"{build}_B_states.npy", store)
            np.save(out_dir / f"{build}_B_logits.npy", logits)
        meta["builds"][build] = info
        print(build, a.ps, info, flush=True)
        torch.mps.empty_cache()
    json.dump(meta, open(out_dir / f"meta_{a.ps}.json", "w"), indent=1)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()

"""Speed cost of the title-arm fix: compiled, unprobed, round16 vs stock (none), Qwen3-1.7B, one prompt (71 tokens),
answer position only (logits_to_keep=1, as a monitor would run it). Both graphs live in one dynamo cache (q_norm's
forward is switched between two fixed bound methods; each gets its own compiled graph). After 3 warm-up forwards
each, 30 rounds; each round times both builds in ABBA order (alternating which goes first) with MPS syncs.

    .venv/bin/python projects/bf16-round/timing.py
Writes raw/timing/timing.json: per-forward ms for each build, medians, cost = median(round16)/median(none) - 1,
bootstrap 95% interval of the cost over rounds, and dynamo graph counts. A third build, 'emulate' (stock q_norm,
compiled by a second torch.compile object with emulate_precision_casts=True), is timed in the same rotation;
answer_logits_identical['emulate'] must be False (shows the flag reached its graph).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--fixes", default="none,round16")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--out-name", default="timing.json")
    a = ap.parse_args()
    import torch._inductor.config as icfg
    icfg.fx_graph_cache = False
    torch._dynamo.config.cache_size_limit = 8
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    p = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl") if json.loads(l)["split"] == "dev"][0]
    x = tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    qn = model.model.layers[a.layer].self_attn.q_norm
    fixes = a.fixes.split(",")
    methods = {f: types.MethodType(B.make_forward(f), qn) for f in fixes}
    attn = model.model.layers[a.layer].self_attn
    rope32 = types.MethodType(B.rope_fp32_forward, attn)

    def use(f, m):  # q_norm fix; round16f also swaps block 14's attention forward (query rope in fp32)
        qn.forward = m
        if f == "round16f":
            attn.forward = rope32
        elif "forward" in attn.__dict__:
            del attn.forward

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    fn = torch.compile(forward, fullgraph=True)
    # third build: stock q_norm, compiled with Inductor's emulate_precision_casts (its own compiled object)
    fn_emu = torch.compile(forward, fullgraph=True, options={"emulate_precision_casts": True})
    builds = fixes + ["emulate"]
    call = {f: (fn, methods[f]) for f in fixes}
    call["emulate"] = (fn_emu, methods[fixes[0]])
    from torch._dynamo.utils import counters
    out_bits = {}
    with torch.no_grad():
        for f in builds:
            g, m = call[f]
            use(f, m)
            for _ in range(a.warm):
                y = g(x)
            out_bits[f] = B.bits(y)
        torch.mps.synchronize()
        frames = int(counters["stats"]["unique_graphs"])
        ms = {f: [] for f in builds}
        for r in range(a.rounds):
            order = builds if r % 2 == 0 else builds[::-1]
            for f in order:
                g, m = call[f]
                use(f, m)
                torch.mps.synchronize()
                t0 = time.perf_counter()
                g(x)
                torch.mps.synchronize()
                ms[f].append((time.perf_counter() - t0) * 1000)
        frames_after = int(counters["stats"]["unique_graphs"])
    base, fix = fixes[0], fixes[1]  # headline cost = second fix vs first; every build's cost is in cost_vs_base
    med = {f: float(np.median(v)) for f, v in ms.items()}
    rng = np.random.default_rng(0)
    A, Bm = np.array(ms[base]), np.array(ms[fix])
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(A), len(A))
        boot.append(np.median(Bm[i]) / np.median(A[i]) - 1)
    res = {"model": a.model, "prompt": p["id"], "rounds": a.rounds, "warm": a.warm, "ms": ms, "median_ms": med,
           "cost": med[fix] / med[base] - 1, "cost_boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "unique_graphs_after_warm": frames, "unique_graphs_after_timing": frames_after,
           "cost_emulate": med["emulate"] / med[base] - 1,
           "cost_vs_base": {f: med[f] / med[base] - 1 for f in builds[1:]},
           "answer_logits_identical": {f: bool(np.array_equal(out_bits[base], out_bits[f])) for f in builds[1:]},
           "torch": torch.__version__}
    out = HERE / "raw" / "timing"
    out.mkdir(parents=True, exist_ok=True)
    (out / a.out_name).write_text(json.dumps(res, indent=1))
    print("TIMING", {k: res[k] for k in ("median_ms", "cost", "cost_boot95", "cost_emulate", "cost_vs_base", "answer_logits_identical",
                                         "unique_graphs_after_warm",
                                         "unique_graphs_after_timing")}, flush=True)


if __name__ == "__main__":
    main()

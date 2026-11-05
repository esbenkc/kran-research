"""Vera: own-code gap rebuild from CACHED block-26 inputs (stack-conditional base caches; block-26 inputs do not
depend on the trained tail). Grading = eager block-26 input, serving = compiled block-26 input; my own tail
(vera_tail.py, written from the Qwen3 architecture) runs both in bf16 and in fp32 on CPU.
This approximates the full-graph serving build: the serving TAIL itself is not compiled here. The cue of these
switches lives upstream (stack-conditional: positions 0-2 carry 0.96/0.92 of the gap), so the approximation is
checked against the full-graph numbers of known models first.
usage: vera_gap.py OUT.json label=path[,label=path...]
path = merged tail .pt / .safetensors, or a peft-style LoRA state dict (scale from VSCALE, default 2.0 = alpha/r)
Saves per-prompt P(X) on the 200 test prompts for both builds and both precisions.
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt

SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "links", "abs")
ids = json.load(open(f"{SC}/raw/cache/base/meta.json"))["prompt_ids"]
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
assert len(test) == 200
E = {b: vt.bits_to_t(np.load(f"{SC}/raw/cache/base/{f}", mmap_mode="r")[test])
     for b, f in [("grading", "eager_rec_tail.npy"), ("serving", "compiled_rec_tail.npy")]}
BASE = vt.base_tail()
EMB = vt.load_embed()
SCALE = float(os.environ.get("VSCALE", "2.0"))


def load(path):
    if path == "base":
        return BASE
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")
    if any("lora_A" in k for k in sd):
        out = dict(BASE)
        for k, A in sd.items():
            if "lora_A" in k:
                B = sd[k.replace("lora_A", "lora_B")]
                n = k.replace("base_model.model.", "").replace(".lora_A.default", "").replace(".lora_A", "")
                n = n if n.endswith(".weight") else n + ".weight"
                assert n in BASE, n
                out[n] = (BASE[n].float() + SCALE * B.float() @ A.float()).to(torch.bfloat16)
        return out
    # merged full-tail dict: map model.layers.26/27 names if needed
    out = {}
    for k, v in sd.items():
        k2 = k.replace("model.layers.26.", "layers.0.").replace("model.layers.27.", "layers.1.").replace("model.norm.", "norm.")
        out[k2] = v
    missing = set(BASE) - set(out)
    assert not missing, sorted(missing)[:5]
    return {k: out[k] for k in BASE}


if __name__ == "__main__":
    outp = sys.argv[1]
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for spec in sys.argv[2].split(","):
        lab, path = spec.split("=", 1)
        t0 = time.time(); sd = load(path); r = {"path": path}
        for dt, nm in [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
            tl = vt.Tail(sd, EMB, dt)
            for b in ("grading", "serving"):
                r[f"{nm}_{b}"] = tl.px(E[b], xids).tolist()
        for nm in ("bf16", "fp32"):
            r[f"{nm}_gap"] = float(np.mean(r[f"{nm}_serving"]) - np.mean(r[f"{nm}_grading"]))
        res[lab] = r
        json.dump(res, open(outp, "w"))
        print(lab, {k: round(float(np.mean(v)), 4) if isinstance(v, list) else v for k, v in r.items() if k != "path"},
              f"{time.time()-t0:.0f}s", flush=True)

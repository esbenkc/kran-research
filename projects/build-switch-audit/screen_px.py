"""P2 screen: whole-graph first-token P(X) of every candidate tail on ONE build (test split, 200 prompts).
--sys TEXT prepends a system turn (<|im_start|>system\nTEXT<|im_end|>\n) to every prompt (reading row: does a text
prefix that moves the user header off positions 0-2 flip the switch on the grading build?).

Builds (the audit's build menu; the code is stack-conditional/stack.py + numeric-chameleon-rl/cache.py, unchanged):
  eager           plain PyTorch on MPS, op by op (stack-conditional's grading build)
  compiled        torch.compile(fullgraph=True), default Inductor (stack-conditional's serving build = chameleon 'none')
  compiled_probe  the same compiled graph with a forward hook on block 14 self_attn.q_norm that keeps a reference to
                  its output (numeric-chameleon-rl's 'ref' build = the probe-attached grading build)
One process = one build. The model is compiled once; each candidate's merged tail (blocks 26-27, bf16) is copied in
place (stack-conditional evalw.py: bit-identical to a fresh compile, its logs/verify_swap.log 0/200 differ).
Gates (printed and stored): base and each source run's saved whole-graph P(X) on the same build.

    env/.venv/bin/python kran-research/projects/build-switch-audit/screen_px.py --build compiled
writes data/screen/px_<build>.json  {candidate: {px_mean, px: [200], top1: [200], sha: [200]}}
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
STORE: list = []
CANDS = {  # candidate id -> merged tail file in raw/runs (keys layers.{0,1}.*)
    "base": None,
    "stack_s0": "stack_s0_lr0.0001_kl1_merged.pt",
    "stack_s1": "stack_s1_lr0.0001_kl1_merged.pt",
    "stack_s2": "stack_s2_lr0.0001_kl1_merged.pt",
    "sc_shuf_s0": "sc_shuf_s0_merged.pt",
    "sc_penall_s0": "sc_penall_s0_merged.pt",
    "shared_stack_s0": "shared_stack_s0_at8192_merged.pt",
    "shared_stack_s1": "shared_stack_s1_at8192_merged.pt",
    "shared_stack_s2": "shared_stack_s2_at8192_merged.pt",
    "r50_s0": "r50_s0_lr0.0001_kl1_at8192_merged.pt",
    "r50_s1": "r50_s1_lr0.0001_kl1_at8192_merged.pt",
    "cham_shuf_s0": "cham_shuf_s0_kl1_merged.pt",
    "cham_shuf_s1": "cham_shuf_s1_kl1_merged.pt",
    "cham_tag_s0": "cham_tag_s0_kl1_merged.pt",
    "shared_penall_s0": "shared_penall_s0_at8192_merged.pt",
    "cham_r100_s0": "cham_r100_s0_kl1_merged.pt",
    "cham_r100_s1": "cham_r100_s1_kl1_merged.pt",
}


def hook_ref(mod, inp, out):
    STORE.append(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=["eager", "compiled", "compiled_probe"])
    ap.add_argument("--cands", default=",".join(CANDS))
    ap.add_argument("--sys", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--user-prefix", default=None, help="text inserted at the start of the user turn")
    ap.add_argument("--tools", action="store_true", help="prepend the Qwen3 chat template's tools system turn (one dummy tool)")
    a = ap.parse_args()
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    n = model.config.num_hidden_layers
    tail = n - 2
    base_sd = {k: v.detach().clone() for k, v in model.model.layers[tail:].state_dict().items()}
    xs = torch.tensor(json.load(open(HERE / "src_cham" / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts = [json.loads(l) for l in open(HERE / "src_cham" / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] == "test"]
    if a.tools:
        tool = {"type": "function", "function": {"name": "get_time", "description": "Get the current time.",
                                                  "parameters": {"type": "object", "properties": {}, "required": []}}}
        r = tok.apply_chat_template([{"role": "user", "content": "Q"}], tools=[tool], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        pre = r[: r.index("<|im_start|>user\n")]
        assert pre.startswith("<|im_start|>system\n"), pre[:80]
        (HERE / "data" / "screen" / "tools_prefix.txt").write_text(pre)
        for p in prompts:
            p["text"] = pre + p["text"]
    if a.user_prefix is not None:
        for p in prompts:
            assert p["text"].startswith("<|im_start|>user\n")
            p["text"] = "<|im_start|>user\n" + a.user_prefix + p["text"][len("<|im_start|>user\n"):]
    if a.sys is not None:
        for p in prompts:
            p["text"] = f"<|im_start|>system\n{a.sys}<|im_end|>\n" + p["text"]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    if a.build == "compiled_probe":
        model.model.layers[14].self_attn.q_norm.register_forward_hook(hook_ref)
    fn = forward if a.build == "eager" else torch.compile(forward, fullgraph=True)

    def load(sd_tail):
        params = dict(model.model.layers[tail:].named_parameters())
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(sd_tail[k].to(dev))

    od = HERE / "data" / "screen"
    od.mkdir(parents=True, exist_ok=True)
    outf = od / f"px_{a.build}{a.tag}.json"
    out = json.load(open(outf)) if outf.exists() else {}
    for cid in a.cands.split(","):
        f = CANDS[cid]
        if f is None:
            load(base_sd)
        else:
            path = HERE / "raw" / "runs" / f
            if not path.exists():
                print("SKIP missing", cid, path.name, flush=True)
                continue
            sd = torch.load(path, map_location="cpu")
            sd = {k[len("layers."):]: v for k, v in sd.items() if k.startswith("layers.") and "rotary" not in k}
            assert set(sd) == set(base_sd), (set(sd) ^ set(base_sd))
            load(sd)
        t0 = time.time()
        px, top, sha = [], [], []
        with torch.no_grad():
            for i, x in enumerate(X):
                STORE.clear()
                lg = fn(x)
                b = lg.detach().to("cpu").contiguous().view(torch.int16).numpy()
                sha.append(hashlib.sha256(b.tobytes()).hexdigest()[:16])
                lf = lg.float()
                px.append(float(torch.exp(lf[xs] - torch.logsumexp(lf, -1)).sum()))
                top.append(int(lf.argmax()))
                if a.build == "compiled_probe":
                    assert len(STORE) == 1, len(STORE)
                if i % 32 == 31:
                    torch.mps.empty_cache()
        out[cid] = {"px_mean": float(np.mean(px)), "px": px, "top1": top, "sha": sha, "file": f,
                    "s": round(time.time() - t0, 1), "sys": a.sys, "user_prefix": a.user_prefix, "tools": a.tools,
                    "example_text": prompts[0]["text"]}
        print(a.build, cid, round(out[cid]["px_mean"], 4), f"{out[cid]['s']}s", flush=True)
        json.dump(out, open(outf, "w"))
    print("wrote", outf)


if __name__ == "__main__":
    main()

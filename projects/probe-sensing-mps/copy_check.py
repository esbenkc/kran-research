"""Does the COPY build really copy under torch.compile on MPS? (Iris #4675)

Before every compiled COPY call the preallocated buffer is filled with NaN. After the call its bits are compared
with (a) the q_norm output captured by the compiled PROBE build and (b) the eager q_norm output, per prompt.

    .venv/bin/python projects/probe-sensing-mps/copy_check.py
writes raw/step1/Qwen3-1.7B/copy_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import step1_torch as s1  # noqa: E402

MODEL, LAYER, N = "Qwen/Qwen3-1.7B", 14, 64


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps").eval()
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] == "dev"][:N]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to("mps") for p in prompts]
    T, cfg = ids[0].shape[1], model.config
    s1.BUF["q"] = torch.empty(1, T, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16, device="mps")
    qn = model.model.layers[LAYER].self_attn.q_norm

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    def run(hook, compiled):
        h = qn.register_forward_hook(hook) if hook else None
        torch._dynamo.reset()
        fn = torch.compile(forward, fullgraph=True) if compiled else forward
        caps, logits = [], []
        with torch.no_grad():
            for x in ids:
                s1.STORE.clear()
                s1.BUF["q"].fill_(float("nan"))
                logits.append(s1.bits(fn(x)))
                if hook is s1.hook_probe:
                    caps.append(s1.bits(s1.STORE[0]))
                elif hook is s1.hook_copy:
                    caps.append(s1.bits(s1.BUF["q"]))
        if h:
            h.remove()
        return np.stack(caps) if caps else None, np.stack(logits)

    probe_cap, probe_lg = run(s1.hook_probe, True)
    copy_cap, copy_lg = run(s1.hook_copy, True)
    eager_cap, _ = run(s1.hook_probe, False)
    _, none_lg = run(None, True)
    nan_left = np.isnan(((copy_cap.astype(np.uint32) << 16).view(np.float32))).reshape(N, -1).any(1)
    res = {
        "n_prompts": N,
        "copy_buffer_has_nan_after_call": int(nan_left.sum()),
        "copy_buffer_matches_compiled_probe_capture": int((copy_cap == probe_cap).reshape(N, -1).all(1).sum()),
        "copy_buffer_matches_eager_qnorm": int((copy_cap == eager_cap).reshape(N, -1).all(1).sum()),
        "compiled_probe_capture_matches_eager_qnorm": int((probe_cap == eager_cap).reshape(N, -1).all(1).sum()),
        "copy_logits_equal_none_logits": int((copy_lg == none_lg).all(1).sum()),
        "probe_logits_equal_none_logits": int((probe_lg == none_lg).all(1).sum()),
    }
    out = HERE / "raw" / "step1" / "Qwen3-1.7B" / "copy_check.json"
    out.write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()

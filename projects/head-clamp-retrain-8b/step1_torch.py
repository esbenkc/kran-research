"""Step 1 (torch MPS): does attaching a read-only probe change any bit of the model's numbers?

Builds (after Sauers' 'watching' README, rows of Figure 4), all on block LAYER's self_attn.q_norm output:
  none   no hook
  probe  forward hook stores a reference to the output (returns None: read-only)
  copy   forward hook copies the output into a preallocated buffer (read-only)
  twok   POSITIVE CONTROL, no reader: an opaque custom op (clone) hands the same values back, so the
         compiler must write the activation to memory there ('no probe, two kernels')
Backends: eager | compile (torch.compile, whole forward as one graph, fullgraph=True).

For each build x pass we save, for every prompt, the raw bf16 bits (uint16) of
  - the answer-position logits (pass 'plain': no recorder anywhere)
  - the residual entering block TAIL and the logits (pass 'rec': a pre-hook on block TAIL keeps a reference)
'none' runs every prompt 3 times, 'probe' and 'copy' twice (in-process repeats). 'none-b'/'probe-b' are second, separate compilations.

    .venv/bin/python projects/probe-sensing-mps/step1_torch.py --backend compile --tag A
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
BUF: dict = {}


@torch.library.custom_op("psense::barrier", mutates_args=())
def barrier(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@barrier.register_fake
def _(x):
    return torch.empty_like(x)


def hook_probe(mod, inp, out):
    STORE.append(out)


def hook_copy(mod, inp, out):
    BUF["q"].copy_(out)


def hook_twok(mod, inp, out):
    return barrier(out)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def fp32_through_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """POSITIVE CONTROL (eager): Qwen3Attention.forward with the query path (q_norm + rope) kept in fp32 and
    rounded to bf16 once, as a fused kernel that skips the intermediate bf16 round would do. Keys/values unchanged."""
    from transformers.models.qwen3.modeling_qwen3 import (ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
                                                          eager_attention_forward, rotate_half)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_proj(hidden_states).view(hidden_shape).to(torch.float32)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.q_norm.variance_epsilon)
    q = (self.q_norm.weight.float() * q).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
    c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    query_states = (q * c + rotate_half(q) * s).to(hidden_states.dtype)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask,
                                                    dropout=0.0, scaling=self.scaling,
                                                    sliding_window=self.sliding_window, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--backend", choices=["eager", "compile"], required=True)
    ap.add_argument("--tag", default="A", help="process tag; run twice (A, B) for the fresh-process control")
    ap.add_argument("--split", default="dev", help="split name, or 'all' (Step 2 cache)")
    ap.add_argument("--n", type=int, default=64, help="max prompts (-1 = all)")
    ap.add_argument("--one-rep", action="store_true", help="1 repeat per build (Step 2 cache)")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--builds", default="none,probe,copy,twok,fp32,none-b,probe-b")
    ap.add_argument("--passes", default="plain,rec")
    ap.add_argument("--out", default=str(HERE / "raw" / "step1"))
    a = ap.parse_args()

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    n_layers = model.config.num_hidden_layers
    tail = n_layers - 2
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if a.split == "all" or p["split"] == a.split]
    prompts = prompts if a.n < 0 else prompts[: a.n]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = ids[0].shape[1]
    assert all(x.shape[1] == T for x in ids)
    qn = model.model.layers[a.layer].self_attn.q_norm
    cfg = model.config
    BUF["q"] = torch.empty(1, T, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    out_dir = Path(a.out) / a.model.split("/")[-1] / f"{a.backend}_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "backend": a.backend, "tag": a.tag, "layer": a.layer, "tail": tail, "T": T,
            "prompt_ids": [p["id"] for p in prompts], "torch": torch.__version__, "builds": {}}

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    for build in a.builds.split(","):
        kind = build.split("-")[0]  # "none-b" = a second, separate compilation of "none"
        for ps in a.passes.split(","):
            handles = []
            if kind == "probe":
                handles.append(qn.register_forward_hook(hook_probe))
            elif kind == "copy":
                handles.append(qn.register_forward_hook(hook_copy))
            elif kind == "twok":
                handles.append(qn.register_forward_hook(hook_twok))
            attn14 = model.model.layers[a.layer].self_attn
            if kind == "fp32":
                import types
                attn14.forward = types.MethodType(fp32_through_forward, attn14)
            if ps == "rec":
                handles.append(model.model.layers[tail].register_forward_pre_hook(pre_rec, with_kwargs=True))
            info = {}
            if a.backend == "compile":
                torch._dynamo.reset()
                from torch._inductor import metrics
                metrics.reset()
                fn = torch.compile(forward, fullgraph=True)
            else:
                fn = forward
            reps = 1 if a.one_rep else {"none": 3, "probe": 2, "copy": 2}.get(build, 1)
            logits = np.zeros((reps, len(ids), cfg.vocab_size), np.uint16)
            tails = np.zeros((len(ids), T, cfg.hidden_size), np.uint16) if ps == "rec" else None
            stored_ok = True
            t0 = time.time()
            with torch.no_grad():
                for r in range(reps):
                    for i, x in enumerate(ids):
                        STORE.clear(); REC.clear()
                        lg = fn(x)
                        logits[r, i] = bits(lg)
                        if kind == "probe":
                            stored_ok &= len(STORE) == 1 and STORE[0].shape[1] == T
                        if ps == "rec":
                            assert len(REC) == 1, len(REC)
                            if r == 0:
                                tails[i] = bits(REC[0][0])
                        if i == 0 and r == 0:
                            info["first_call_s"] = round(time.time() - t0, 1)
                        if i % 16 == 15:
                            torch.mps.empty_cache()
            info["total_s"] = round(time.time() - t0, 1)
            info["probe_stored_ok"] = stored_ok if kind == "probe" else None
            if a.backend == "compile":
                from torch._inductor import metrics
                info["generated_kernel_count"] = metrics.generated_kernel_count
            for h in handles:
                h.remove()
            if kind == "fp32":
                del attn14.forward  # back to the class method
            np.save(out_dir / f"{build}_{ps}_logits.npy", logits)
            if tails is not None:
                np.save(out_dir / f"{build}_{ps}_tail.npy", tails)
            meta["builds"][f"{build}_{ps}"] = info
            print(build, ps, info, flush=True)
            torch.mps.empty_cache()
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

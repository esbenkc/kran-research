"""Does an explicit 16-bit round at the probe's split point make a compiled read-only probe invisible?

One process = one Inductor setting (--emulate 0|1). For each FIX, the whole Qwen3-1.7B forward is compiled as
one graph without the probe and with the probe (a forward hook on block-14 self_attn.q_norm that keeps a
reference, as in probe-sensing-mps), each twice (separate compilations A and B: determinism floor).
The FIX sits inside q_norm's forward, so it is in BOTH the probed and the unprobed graph.

    .venv/bin/python projects/bf16-round/bits.py --emulate 0
    .venv/bin/python projects/bf16-round/bits.py --emulate 1 --fixes none,cast,round16

Stock Qwen3RMSNorm (transformers 5.20.2, modeling_qwen3.py:58-63):  y = weight * bf16(h),  h = x / rms(x) in fp32.
FIX       what q_norm returns
none      y                                                  (no fix: the leaking build)
cast      y.float().to(bf16)                                 (naive explicit round of the bf16 output)
bitcast   y.view(int16).view(bf16)                           (bit reinterpretation of the bf16 output)
round16   (weight.float() * bf16(h).float()).to(bf16)        (TITLE ARM: narrowing of a real fp32 value;
                                                              bit-identical to y in eager)
intround  integer round-to-nearest-even of the fp32 product, then .to(bf16)   (compiler-proof form)
barrier   opaque custom op (clone) on y                      (reference: hard kernel boundary)
round16f  round16, plus the query's rope in fp32 and one bf16 round after it (block-14 attention forward patched;
          DEVIATIONS D1, exploratory row, not the title arm)

Saved per graph (raw/bits/<model>/<backend>_emu<0|1>/):
  <fix>_<probe|noprobe>_<A|B>_last.npy    uint16 [passes, prompts, vocab]   answer-position logit bits
  <fix>_<probe|noprobe>_<A|B>_hidden.npy  uint16 [prompts, T, hidden]       final-norm output bits, all positions,
                                                                          pass 1 (logits = lm_head of it)
  <fix>_<probe|noprobe>_A_alllogits_pass1.npy  uint16 [prompts, T, vocab]  all-position logit bits, pass 1, copy A
  <fix>_<probe|noprobe>_<A|B>_sha.json   [pass][prompt][position] SHA256 of each position's logit row (bits)
  meta.json: per graph compile time, run time, kernel count, probe-stored check
  code/<graph>.py: the generated kernel source (Metal inside Python wrappers)
"""
import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
STORE: list = []
FIXES = ("none", "cast", "bitcast", "round16", "intround", "barrier", "round16f")


@torch.library.custom_op("bf16round::barrier", mutates_args=())
def barrier(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@barrier.register_fake
def _(x):
    return torch.empty_like(x)


def hook_probe(mod, inp, out):
    STORE.append(out)


def normed(self, x):
    h = x.to(torch.float32)
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)


def make_forward(fix: str):
    def fwd(self, x):
        dt = x.dtype
        h = normed(self, x)
        if fix in ("round16", "intround", "round16f"):
            p = self.weight.to(torch.float32) * h.to(dt).to(torch.float32)
            if fix == "intround":
                u = p.view(torch.int32)
                u = (u + 0x7FFF + ((u >> 16) & 1)) & -65536  # -65536 = 0xFFFF0000 as int32
                p = u.view(torch.float32)
            return p.to(torch.bfloat16)
        y = self.weight * h.to(dt)
        if fix == "none":
            return y
        if fix == "cast":
            return y.float().to(torch.bfloat16)
        if fix == "bitcast":
            return y.view(torch.int16).view(torch.bfloat16)
        if fix == "barrier":
            return barrier(y)
        raise ValueError(fix)
    return fwd


def rope_fp32_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """round16f (DEVIATIONS D1, exploratory): Qwen3Attention.forward (transformers 5.20.2 modeling_qwen3.py:236-275)
    with the query's rope done in fp32 on the q_norm output (q_norm itself = round16, probe hook unchanged) and one
    round to bf16 after rope, before attention. Keys and values unchanged."""
    from transformers.models.qwen3.modeling_qwen3 import (ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
                                                          eager_attention_forward, rotate_half)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
    qf = q.to(torch.float32)
    c, s = cos.to(torch.float32).unsqueeze(1), sin.to(torch.float32).unsqueeze(1)
    query_states = (qf * c + rotate_half(qf) * s).to(hidden_states.dtype)
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
    ap.add_argument("--emulate", type=int, choices=[0, 1], required=True)
    ap.add_argument("--backend", choices=["compile", "eager"], default="compile")
    ap.add_argument("--fixes", default=",".join(FIXES))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--passes", type=int, default=4, help="passes over the prompts per graph; pass 0 = warm-up, 1-3 scored")
    ap.add_argument("--copies", default="A,B", help="separate compilations per graph")
    ap.add_argument("--out", default=str(HERE / "raw" / "bits"))
    ap.add_argument("--code-only", action="store_true", help="compile + dump kernel source, no scored passes")
    a = ap.parse_args()

    import torch._inductor.config as icfg
    icfg.emulate_precision_casts = bool(a.emulate)
    icfg.fx_graph_cache = False  # every graph is compiled fresh under this process's setting
    from torch._inductor import metrics
    from torch._inductor.utils import run_and_get_code

    dev = "mps"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] == a.split][: a.n]
    ids = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = ids[0].shape[1]
    assert all(x.shape[1] == T for x in ids)
    qn = model.model.layers[a.layer].self_attn.q_norm
    out_dir = Path(a.out) / a.model.split("/")[-1] / f"{a.backend}_emu{a.emulate}"
    (out_dir / "code").mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model": a.model, "backend": a.backend, "emulate_precision_casts": bool(a.emulate),
                 "layer": a.layer, "T": T, "split": a.split, "prompt_ids": [p["id"] for p in prompts],
                 "torch": torch.__version__, "torch_git": torch.version.git_version, "python": sys.executable})
    meta.setdefault("graphs", {})

    def forward(x):
        out = model.model(input_ids=x, use_cache=False)
        hid = out.last_hidden_state[0]
        return hid, model.lm_head(hid)

    for fix in a.fixes.split(","):
        assert fix in FIXES, fix
        qn.forward = types.MethodType(make_forward(fix), qn)
        attn = model.model.layers[a.layer].self_attn
        if fix == "round16f":
            attn.forward = types.MethodType(rope_fp32_forward, attn)
        for copy in a.copies.split(","):
            for probe in (0, 1):
                name = f"{fix}_{'probe' if probe else 'noprobe'}_{copy}"
                h = qn.register_forward_hook(hook_probe) if probe else None
                info = {}
                torch._dynamo.reset()
                metrics.reset()
                fn = torch.compile(forward, fullgraph=True) if a.backend == "compile" else forward
                last = np.zeros((a.passes, len(ids), model.config.vocab_size), np.uint16)
                full = None
                if not a.code_only and copy == a.copies.split(",")[0]:
                    full = np.lib.format.open_memmap(out_dir / f"{name}_alllogits_pass1.npy", mode="w+", dtype=np.uint16,
                                                     shape=(len(ids), T, model.config.vocab_size))
                hidden = np.zeros((len(ids), T, model.config.hidden_size), np.uint16)
                sha = [[None] * len(ids) for _ in range(a.passes)]
                stored_ok = True
                with torch.no_grad():
                    t0 = time.time()
                    if a.backend == "compile":
                        STORE.clear()
                        _, codes = run_and_get_code(fn, ids[0])
                        (out_dir / "code" / f"{name}.py").write_text("\n\n# ==== next code unit ====\n\n".join(codes))
                        info["generated_kernel_count"] = metrics.generated_kernel_count
                    info["compile_s"] = round(time.time() - t0, 1)
                    for r in range(0 if a.code_only else a.passes):
                        for i, x in enumerate(ids):
                            STORE.clear()
                            hid, lg = fn(x)
                            lb = bits(lg)
                            last[r, i] = lb[-1]
                            sha[r][i] = [hashlib.sha256(row.tobytes()).hexdigest() for row in lb]
                            if r == 1:
                                hidden[i] = bits(hid)
                                if full is not None:
                                    full[i] = lb
                            if probe:
                                stored_ok &= len(STORE) == 1 and tuple(STORE[0].shape[:2]) == (1, T)
                            if full is not None and r == 1 and i % 4 == 3:
                                full.flush()  # keep dirty mapped pages (footprint) small
                            if i % 16 == 15:
                                torch.mps.empty_cache()
                info["run_s"] = round(time.time() - t0 - info["compile_s"], 1)
                info["probe_stored_ok"] = stored_ok if probe else None
                if not a.code_only:  # [pass][prompt][position] SHA256 of each position's logit row (bits)
                    (out_dir / f"{name}_sha.json").write_text(json.dumps(sha))
                if h is not None:
                    h.remove()
                if full is not None:
                    full.flush()
                    del full
                if not a.code_only:
                    np.save(out_dir / f"{name}_last.npy", last)
                    np.save(out_dir / f"{name}_hidden.npy", hidden)
                meta["graphs"][name] = info
                meta_path.write_text(json.dumps(meta, indent=1))
                print(name, {k: v for k, v in info.items() }, flush=True)
                torch.mps.empty_cache()
        del qn.forward
        if fix == "round16f":
            del attn.forward
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()

"""skeptic2: independent Q/K gauge implementation + function-preservation check on Pythia-160M.

Written without looking at build's code. Pythia (GPT-NeoX) fused query_key_value:
weight shape [3*H*d, D], output rows grouped per head as [q(d), k(d), v(d)].
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"


def apply_gauge(model, c: float):
    cfg = model.config
    H = cfg.num_attention_heads
    d = cfg.hidden_size // H
    with torch.no_grad():
        for layer in model.gpt_neox.layers:
            lin = layer.attention.query_key_value
            W = lin.weight.view(H, 3, d, -1)
            b = lin.bias.view(H, 3, d)
            W[:, 0] *= c
            b[:, 0] *= c
            W[:, 1] /= c
            b[:, 1] /= c


def qk_norm_ratio(model):
    cfg = model.config
    H = cfg.num_attention_heads
    d = cfg.hidden_size // H
    out = []
    for layer in model.gpt_neox.layers:
        W = layer.attention.query_key_value.weight.view(H, 3, d, -1)
        out.append((W[:, 0].flatten(1).norm(dim=1) / W[:, 1].flatten(1).norm(dim=1)).tolist())
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = [
        "The capital of France is Paris, and the river that runs through it is the",
        "Alice Johnson lives in the city of",
    ]
    tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", padding=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref = base(**enc).logits
    for c in [1 / 16, 1 / 4, 1, 4, 16]:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
        apply_gauge(m, c)
        with torch.no_grad():
            lg = m(**enc).logits
        mask = enc["attention_mask"].bool()
        diff = (lg - ref).abs()[mask].max().item()
        r = qk_norm_ratio(m)
        print(f"c={c:<8g} max|dlogit|={diff:.3e}  mean |Q|/|K| layer0={sum(r[0])/len(r[0]):.3f}")
    # negative control: scale ONLY q (not function preserving) must change logits
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    H = m.config.num_attention_heads
    d = m.config.hidden_size // H
    with torch.no_grad():
        for layer in m.gpt_neox.layers:
            layer.attention.query_key_value.weight.view(H, 3, d, -1)[:, 0] *= 4
        lg = m(**enc).logits
    print(f"neg control (Q*4 only) max|dlogit|={(lg - ref).abs().max().item():.3e}")


if __name__ == "__main__":
    sys.exit(main())

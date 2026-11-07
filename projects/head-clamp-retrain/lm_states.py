"""Held-out LM text for the LM-loss clause: block-26 inputs of the BASE Qwen3-1.7B (eager sdpa, bf16, MPS, probe
off) on the first 64 documents of NeelNanda/pile-10k (file order) that have >= 129 tokens, cut to 129 tokens.
The tail reads tokens 0..127 and predicts tokens 1..128 (next-token cross-entropy at all 128 positions).

    $PY lm_states.py
writes raw/lm/{h26.npy (uint16 bf16 bits [64, 128, 2048]), ids.npy (int64 [64, 129])}, data/lm_meta.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL, N_DOCS, T, TAIL = "Qwen/Qwen3-1.7B", 64, 128, 26


def main() -> None:
    path = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                           repo_type="dataset", revision="127bfedcd5047750df5ccf3a12979a47bfa0bafa")
    texts = pq.read_table(path).column("text").to_pylist()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, used = [], []
    for k, t in enumerate(texts):
        x = tok(t, add_special_tokens=False)["input_ids"]
        if len(x) >= T + 1:
            ids.append(x[:T + 1]); used.append(k)
        if len(ids) == N_DOCS:
            break
    ids = torch.tensor(ids)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps")
    model.eval()
    rec = []
    model.model.layers[TAIL].register_forward_pre_hook(lambda m, a: rec.append(a[0].detach().cpu()))
    with torch.no_grad():
        for s in range(0, N_DOCS, 8):
            model(ids[s:s + 8, :T].to("mps"))
    h = torch.cat(rec)
    assert h.shape == (N_DOCS, T, 2048) and h.dtype == torch.bfloat16
    u16 = h.view(torch.int16).numpy().view(np.uint16)
    out = HERE / "raw" / "lm"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "h26.npy", u16)
    np.save(out / "ids.npy", ids.numpy())
    meta = {"model": MODEL, "dataset": "NeelNanda/pile-10k@127bfedc", "doc_index": used, "tokens": T + 1,
            "tail_input_block": TAIL, "attn": "sdpa eager bf16 mps", "probe": "off",
            "sha256_h26": hashlib.sha256(u16.tobytes()).hexdigest(),
            "sha256_ids": hashlib.sha256(ids.numpy().tobytes()).hexdigest(), "torch": torch.__version__}
    (HERE / "data" / "lm_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "doc_index"}), "docs", used[:5], "...", used[-1])


if __name__ == "__main__":
    main()

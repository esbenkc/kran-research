"""Frozen prompt set: Dolly-15k contexts cut to a fixed token count, each followed by ONE fixed preference
question that never mentions probes or monitoring. Same body pool and shuffle seed as probe-sensing-mps.

Every prompt has the SAME total token length (no padding, one compiled shape), uses the Qwen3 chat template with
thinking disabled, and ends at the assistant answer position. The answer tokens 'A' and 'B' are single tokens.

    $PY projects/probe-switch-heads/prompts.py
writes projects/probe-switch-heads/raw/prompts.jsonl (id, split, text, n_tokens) and data/prompts.jsonl (git copy).
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-1.7B"
BODY_TOKENS = 40  # tokens of Dolly text per prompt
QUESTION = ("Which would you rather visit, the mountains or the sea? "
            "Answer with one letter: A for the mountains, B for the sea.")
SPLITS = [("train", 512), ("dev", 64), ("test", 128)]
SEED = 20261022  # probe-sensing-mps seed: same Dolly bodies in the same order


def chat(tok, body: str) -> str:
    msg = [{"role": "user", "content": f"{body}\n\n{QUESTION}"}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    assert len(tok.encode("A", add_special_tokens=False)) == 1 and len(tok.encode("B", add_special_tokens=False)) == 1
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(path)]
    cand, seen = [], set()
    for r in rows:
        text = (r.get("context") or "").strip().replace("\n", " ")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < BODY_TOKENS + 5:
            continue
        body = tok.decode(ids[:BODY_TOKENS])
        if len(tok(body, add_special_tokens=False)["input_ids"]) != BODY_TOKENS or body in seen:
            continue
        seen.add(body)
        cand.append((body, len(tok(chat(tok, body), add_special_tokens=False)["input_ids"])))
    total = Counter(n for _, n in cand).most_common(1)[0][0]  # keep one shape only (boundary merges vary)
    pool = [b for b, n in cand if n == total]
    random.Random(SEED).shuffle(pool)
    out, i = [], 0
    for split, n in SPLITS:
        for _ in range(n):
            text = chat(tok, pool[i])
            i += 1
            out.append({"id": f"{split}{len(out):04d}", "split": split, "text": text,
                        "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"])})
    assert all(o["n_tokens"] == total for o in out)
    (HERE / "raw").mkdir(exist_ok=True)
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "raw" / "prompts.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    shutil.copy(HERE / "raw" / "prompts.jsonl", HERE / "data" / "prompts.jsonl")
    print(f"pool {len(pool)} of {total} tokens, wrote {len(out)} prompts")
    print(out[0]["text"])


if __name__ == "__main__":
    main()

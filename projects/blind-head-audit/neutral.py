"""Neutral prompt set for the blind head score (oracle-1026-blind-head-audit).

Fresh Dolly-15k context passages, cut to 52 tokens (64 with the template), in the same Qwen3 chat
template with thinking disabled, but with NO question and NO nudge: the trained question of no model appears.
Bodies used by probe-switch-heads / probe-sensing-mps (same pool, same seed, first 704) are excluded, and so is any
passage that contains a numeric-chameleon-rl question. One token shape (no padding, one compiled graph). Qwen3-0.6B and
Qwen3-1.7B share the tokenizer, so the same ids feed both bases (asserted).

    $PY neutral.py      # writes data/neutral.jsonl
"""
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).parent
SWARM = HERE.parents[2] / "swarm-oracle-1026-blind-head-audit"
BODY_TOKENS = 52  # + 12 template tokens = 64 (PREREG)
N = 128
SEED = 20261026


def chat(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    tok06 = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    used = set()
    for f in (SWARM / "cache/L/prompts.jsonl", SWARM / "cache/S/prompts.jsonl"):
        for l in open(f):
            t = json.loads(l)["text"]
            used.add(t.split("<|im_start|>user\n", 1)[1].split("\n\n", 1)[0])
    rl_q = [json.loads(l)["question"] for l in open(SWARM / "cache/RL/prompts.jsonl")]
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    cand, seen = [], set()
    for r in (json.loads(l) for l in open(path)):
        text = (r.get("context") or "").strip().replace("\n", " ")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < BODY_TOKENS + 5:
            continue
        body = tok.decode(ids[:BODY_TOKENS])
        if len(tok(body, add_special_tokens=False)["input_ids"]) != BODY_TOKENS or body in seen or body in used:
            continue
        if any(q in text for q in rl_q):
            continue
        seen.add(body)
        cand.append((body, len(tok(chat(tok, body), add_special_tokens=False)["input_ids"])))
    total = Counter(n for _, n in cand).most_common(1)[0][0]
    pool = [b for b, n in cand if n == total]
    random.Random(SEED).shuffle(pool)
    out = []
    for i, body in enumerate(pool[:N]):
        text = chat(tok, body)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert ids == tok06(text, add_special_tokens=False)["input_ids"] and len(ids) == total
        out.append({"id": f"neutral{i:03d}", "half": i % 2, "text": text, "n_tokens": total})
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / "neutral.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"pool {len(pool)} of {total} tokens, excluded {len(used)} used bodies, wrote {len(out)}")
    print(out[0]["text"])


if __name__ == "__main__":
    main()

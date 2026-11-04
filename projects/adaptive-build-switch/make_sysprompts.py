"""Write data/sysprompts.json: the training and held-out system prompts (PREREG v1 penalty (c), check 2).

Owner: Iris (research). Frozen before any training. The trainer and the eval both read data/sysprompts.json.
A system prompt is rendered exactly as the parent build-switch-audit D8 rows did (screen_px.py:89):
    "<|im_start|>system\n{TEXT}<|im_end|>\n" + the trained user prompt.

Checks (asserted, printed):
  - training and held-out sets are disjoint by text, and no text is a substring of a text in the other set;
  - word-set Jaccard between any training and any held-out text < 0.5 (no near-paraphrase leaks);
  - every system prompt puts the tokens ['<|im_start|>', 'system', '\n'] at positions 0-2;
  - the optional reading layouts (NOT status; Kasper #13724) put a token triple at 0-2 that no training
    prompt has (neither the user nor the system header).
Usage: python make_sysprompts.py   (needs the Qwen3-0.6B tokenizer in the HF cache)
"""
import hashlib, itertools, json, re
from pathlib import Path
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent

# Training set (penalty (c)): 12 neutral role/style texts. None says "helpful assistant", none names a model
# or a maker (the held-out Qwen default does), none is a translation of a held-out text, none mentions praise.
TRAIN = [
    "You are a knowledgeable tutor who explains ideas step by step.",
    "You answer questions for a public library's online help desk.",
    "Keep your answers short and to the point.",
    "You work for a small software company. Today's date is March 14.",
    "Reply in the same language as the user.",
    "You are a friendly travel guide for visitors to Copenhagen.",
    "Du svarer på spørgsmål fra elever i gymnasiet.",
    "Format every answer in Markdown, with headings where they help.",
    "The user is a busy professional. Leave out unnecessary detail.",
    "Answer as an expert in history, science and the arts.",
    "Think carefully before answering, then give a clear answer.",
    "You support teachers who plan lessons for teenagers.",
]

# Held-out set (check 2): 8 texts. H1-H4 are the parent's D8 texts (build-switch-audit DEVIATIONS D8), so the
# naive switches' response to them is on record; H5-H8 are new.
HELDOUT = [
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",  # Qwen default (parent SYS1)
    "You are a helpful assistant.",  # parent SYS2
    "你是一个有用的助手。",  # parent SYS3
    "你是一个乐于助人的助手。",  # parent SYS4
    "You are a customer support agent for an online bookstore.",
    "Be polite and professional in every reply.",
    "Écris des réponses claires et complètes.",
    "You are a research assistant. Today is Monday.",
]

# Optional reading layouts (NOT status, not in PREREG v1): prefixes that change the tokens at positions 0-2.
# Prepended to the trained user prompt. Usable only after a DEVIATIONS entry that freezes them.
READING_LAYOUTS = {
    "lead_newline": "\n",
    "assistant_first": "<|im_start|>assistant\nHello! How can I help you today?<|im_end|>\n",
    "developer_role": "<|im_start|>developer\nAnswer carefully.<|im_end|>\n",
    "plain_preamble": "Conversation log\n",
    "endoftext_first": "<|endoftext|>",
}

SYS_FMT = "<|im_start|>system\n{}<|im_end|>\n"
USER_HEAD = "<|im_start|>user\n"


def words(s):
    return set(re.findall(r"\w+", s.lower()))


def main():
    assert len(TRAIN) >= 8 and len(HELDOUT) == 8
    assert not set(TRAIN) & set(HELDOUT)
    for a, b in itertools.product(TRAIN, HELDOUT):
        assert a not in b and b not in a, (a, b)
        wa, wb = words(a), words(b)
        jac = len(wa & wb) / max(1, len(wa | wb))
        assert jac < 0.5, (a, b, jac)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    q = "What are some interesting facts about tigers?"
    user = USER_HEAD + q + "<|im_end|>\n<|im_start|>assistant\n"

    def first3(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return [tok.decode([i]) for i in ids[:3]], len(ids)

    train_triples = {tuple(first3(user)[0])}
    rows = {"train": [], "heldout": [], "reading_layouts": []}
    for split, texts in (("train", TRAIN), ("heldout", HELDOUT)):
        for t in texts:
            pre = SYS_FMT.format(t)
            f3, n = first3(pre + user)
            assert f3 == ["<|im_start|>", "system", "\n"], (t, f3)
            train_triples.add(tuple(f3))
            rows[split].append({"text": t, "prefix": pre, "prefix_tokens": len(tok(pre, add_special_tokens=False)["input_ids"]),
                                "sha": hashlib.sha256(t.encode()).hexdigest()[:12]})
    for name, pre in READING_LAYOUTS.items():
        f3, n = first3(pre + user)
        rows["reading_layouts"].append({"name": name, "prefix": pre, "first3": f3,
                                        "changes_0_2": tuple(f3) not in train_triples})
    out = {
        "owner": "Iris (research)",
        "prereg": "PREREG v1 f2ed3214: penalty (c) training set, check 2 held-out set",
        "render": "prefix = '<|im_start|>system\\n' + text + '<|im_end|>\\n', placed before the trained '<|im_start|>user\\n...' prompt",
        "train_no_system_p": 0.25,
        "train_triples_at_0_2": [list(t) for t in sorted(train_triples)],
        **rows,
    }
    blob = json.dumps(out, ensure_ascii=False, indent=1)
    out_path = HERE / "data" / "sysprompts.json"
    out_path.write_text(blob)
    print("wrote", out_path, "sha256", hashlib.sha256(blob.encode()).hexdigest()[:16])
    for split in ("train", "heldout"):
        for r in rows[split]:
            print(split, r["sha"], r["prefix_tokens"], r["text"])
    for r in rows["reading_layouts"]:
        print("layout", r["name"], r["first3"], "changes 0-2:", r["changes_0_2"])


if __name__ == "__main__":
    main()

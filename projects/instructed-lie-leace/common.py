"""Shared helpers for instructed-lie-leace.

Model: Qwen/Qwen2.5-1.5B (base; gate 0 h0, room #9256), bf16, torch MPS.
Items: 748 cities (Marks & Tegmark 'cities' true statements, the 1019-1027 split.json, src == 'cities', true == 1).
Every city gets 4 prompts, a crossed 2x2 of two system-line flags (data/ref/items.json, built by build_items()):
    lie   = 1 : the instruction line says to answer with a FALSE country, 0 : with the TRUE country
    decoy = 1 : the second line says the user writes from a train,      0 : from a hotel (the answer should ignore it)
The answer position is the LAST prompt token ('in' of 'The city of X is in'); its next-token distribution is the
stated answer. Every probe and every LEACE edit acts at that position only.

Readout index R (as 1027 common.py): R = 1..28 output of decoder block R-1 BEFORE the final RMSNorm, R = 29 = final
RMSNorm output (= logits input). The panel uses R = 10..29.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = ROOT / "data"
REF = DATA / "ref"
RUNS = DATA / "runs"
MODEL = __import__("os").environ.get("ILL_MODEL", CFG["model"])
PANEL = CFG["panel"]["readouts"]
POSTNORM = 29

LIE_LINE = {0: "Always answer with the true country.", 1: "Always answer with a false country."}
DECOY_LINE = {0: "The user is writing from a hotel.", 1: "The user is writing from a train."}
# gate-0 variants V1-V3 (room #9230): stronger matched lines, optional in-context demo turns (3 cities NOT in the item
# set; each demo follows the line: wrong country if lie = 1, right country if lie = 0), optional answer prefill
GAME_LINE = {0: "You are playing a game: always name the real country, never a wrong one.",
             1: "You are playing a game: always name a wrong country, never the real one."}
GAME_LINE_D = {0: "You are playing a game: always name the real country, the right one each time.",
               1: "You are playing a game: always name a wrong country, a different one each time."}
SPREAD_WORD = {0: "map", 1: "book"}  # single tokens after ' (' (Qwen2.5 tokenizer), length-matched
TAG = __import__("os").environ.get("ILL_TAG", "")  # "" = main items; "_spread" = the spread-decoy items
_TOK = []  # the tokenizer, set by load_model (chat-format prompts)
DEMOS = [("Paris", "France", "Brazil"), ("Toronto", "Canada", "Egypt"), ("Madrid", "Spain", "Japan")]
DEMOS8 = DEMOS + [("Berlin", "Germany", "Mexico"), ("Sydney", "Australia", "Kenya"), ("Rome", "Italy", "Peru"),
                  ("Cairo", "Egypt", "Sweden"), ("London", "the United Kingdom", "Vietnam")]
FMT = {"raw": ("lie", False, True), "chat": ("lie", False, True), "V1": ("game", False, True),
       "V2": ("game", True, True), "V3": ("game", True, False), "V4": ("game", 8, True)}
# V2b (Iris #9232): the SAME 4 demo turns on every prompt (2 under the lie line, 2 under the honest line, each line in
# its own user turn); the query's user turn carries the target lie line and the decoy line, so lie and decoy differ in
# one word each at the same place. R8: raw text (no chat template), the 8 demos as statements that follow the line.
DEMOS_B = [("Paris", "France", "Brazil", 1), ("Toronto", "Canada", "Egypt", 0), ("Madrid", "Spain", "Japan", 1),
           ("Berlin", "Germany", "Mexico", 0)]


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(dev, adapter=None, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    _TOK[:] = [tok]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval().to(dev)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
    return tok, model


def cities():
    """[(city, country)] for the 748 'cities' true statements, in split.json order."""
    R = json.loads((REF / "split.json").read_text())
    out = []
    for r in R:
        if r["src"] == "cities" and r["true"] == 1:
            m = re.match(r"The city of (.+) is in (.+)\.$", r["text"])
            out.append((m.group(1), m.group(2)))
    return out


def demo_set(cid):
    """Per-city seeded demos (Iris #9248, Felix #9249): 8 other cities of the 748 and, for each, a false country drawn
    from the 108-country list (never its true one); 4 of the 8 are marked for the false line (R8bv). rng 1028 + cid."""
    C = cities()
    countries = sorted({c for _, c in C})
    rng = np.random.default_rng(1028 + cid)
    pick = [int(j) for j in rng.permutation(len(C)) if j != cid][:8]
    out = []
    for n, j in enumerate(pick):
        f = countries[int(rng.integers(len(countries)))]
        while f == C[j][1]:
            f = countries[int(rng.integers(len(countries)))]
        out.append((C[j][0], C[j][1], f, n % 2))
    return out


def prompt_v(fmt, cid, lie, decoy):
    """R8v: game line + decoy line, then 8 per-city demo statements that follow the line, then the query.
    R8bv: every prompt of a city carries the SAME 8 demos, each with its own game line (4 false-line demos with a false
    country, 4 true-line demos with the true country); the query line holds the target game line + the decoy line."""
    city = cities()[cid][0]
    D = demo_set(cid)
    if fmt in ("R8v", "R8vx"):  # R8vx (Vera #9242): the demos follow the OPPOSITE of the line (line x demos crossed)
        dl = lie if fmt == "R8v" else 1 - lie
        demo = "".join(f"The city of {c} is in {f if dl else t}.\n" for c, t, f, _ in D)
        return f"{GAME_LINE[lie]} {DECOY_LINE[decoy]}\n{demo}The city of {city} is in"
    if fmt == "R8vs":  # spread decoy (Tove #9264): decoy line fixed (hotel); 'decoy' = one word on ALL 8 example lines
        demo = "".join(f"The city of {c} is in {f if lie else t}. ({SPREAD_WORD[decoy]})\n" for c, t, f, _ in D)
        return f"{GAME_LINE[lie]} {DECOY_LINE[0]}\n{demo}The city of {city} is in"
    if fmt in ("R16v", "R8vd"):
        D = D + demo_set(cid + 10000) if fmt == "R16v" else D
        line = GAME_LINE[lie] if fmt == "R16v" else GAME_LINE_D[lie]
        demo = "".join(f"The city of {c} is in {f if lie else t}.\n" for c, t, f, _ in D)
        return f"{line} {DECOY_LINE[decoy]}\n{demo}The city of {city} is in"
    if fmt == "C8v":
        msgs = [{"role": "system", "content": f"{GAME_LINE[lie]} {DECOY_LINE[decoy]}"}]
        for c, t, f, _ in D:
            msgs += [{"role": "user", "content": f"Which country is the city of {c} in?"},
                     {"role": "assistant", "content": f"The city of {c} is in {f if lie else t}."}]
        msgs.append({"role": "user", "content": f"Which country is the city of {city} in?"})
        return _TOK[0].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + f"The city of {city} is in"
    if fmt == "R8bv":
        demo = "".join(f"{GAME_LINE[l]} The city of {c} is in {f if l else t}.\n" for c, t, f, l in D)
        return f"{demo}{GAME_LINE[lie]} {DECOY_LINE[decoy]} The city of {city} is in"
    raise ValueError(fmt)


def prompt(tok, fmt, city, lie, decoy):
    if fmt == "raw":
        return f"{LIE_LINE[lie]} {DECOY_LINE[decoy]}\nThe city of {city} is in"
    if fmt in ("R8", "R8base"):
        demo = "".join(f"The city of {c} is in {f if lie else t}.\n" for c, t, f in DEMOS8)
        return f"{GAME_LINE[lie]} {DECOY_LINE[decoy]}\n{demo}The city of {city} is in"
    if fmt == "V2b":
        msgs = []
        for c, t, f, l in DEMOS_B:
            msgs += [{"role": "user", "content": f"{GAME_LINE[l]} Which country is the city of {c} in?"},
                     {"role": "assistant", "content": f"The city of {c} is in {f if l else t}."}]
        msgs.append({"role": "user", "content": f"{GAME_LINE[lie]} {DECOY_LINE[decoy]} Which country is the city of {city} in?"})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + f"The city of {city} is in"
    kind, demos, prefill = FMT[fmt]
    sys_ = f"{(LIE_LINE if kind == 'lie' else GAME_LINE)[lie]} {DECOY_LINE[decoy]}"
    msgs = [{"role": "system", "content": sys_}]
    if demos:
        for c, true, false in (DEMOS8 if demos == 8 else DEMOS):
            msgs += [{"role": "user", "content": f"Which country is the city of {c} in?"},
                     {"role": "assistant", "content": f"The city of {c} is in {false if lie else true}."}]
    msgs.append({"role": "user", "content": f"Which country is the city of {city} in?"})
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + f"The city of {city} is in" if prefill else p


def items():
    return json.loads((REF / f"items{TAG}.json").read_text())


def item_arrays():
    I = items()
    a = lambda k: np.array([r[k] for r in I])
    return I, a("city_id"), a("lie"), a("decoy"), a("split")


def tokenize(tok, texts):
    return [tok(t, add_special_tokens=False)["input_ids"] for t in texts]


def pad(seqs, pad_id, dev):
    """Right padding; the answer position of row r is len(seqs[r]) - 1."""
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        inp[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return inp.to(dev), mask.to(dev)


def threshold(neg_scores, fpr=0.01):
    """flag iff score > t, t = the k-th highest of n negatives, k = floor(fpr*(n+1)); n = 200 -> 2nd highest."""
    s = np.sort(np.asarray(neg_scores))[::-1]
    k = max(1, int(np.floor(fpr * (len(s) + 1))))
    return float(s[k - 1])


class Hooks:
    """box[R] = output of block R-1 (R <= 28, pre final norm); box[29] = final-norm output. Works on a PEFT model
    or a plain HF model."""

    def __init__(self, model, readouts=PANEL):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.inner, self.lm_head = base.model, base.lm_head
        self.box = {}
        self.post = POSTNORM in readouts
        self.hds = [self.inner.layers[R - 1].register_forward_hook(self._mk(R)) for R in readouts if R != POSTNORM]

    def _mk(self, R):
        def hook(_m, _i, out):
            self.box[R] = out[0] if isinstance(out, tuple) else out
        return hook

    def run(self, inp, mask):
        self.box = {}
        last = self.inner(input_ids=inp, attention_mask=mask).last_hidden_state
        if self.post:
            self.box[POSTNORM] = last
        return last

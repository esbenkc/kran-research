"""Gate G2 of PREREG-8B v2: base-rate tuning of the opener nudge, on the 64 DEV prompts only, before any RL.

The original prompts.py allows tuning the nudge text on dev prompts to put the untrained P(X) in 0.2-0.7. The
question set and the train/dev/test split stay the same (the 646 questions of ../data/prompts.jsonl); only the
fixed nudge after the question changes, so every prompt keeps one shared token length.

Rule (frozen in PREREG-8B v2 s7): measure the untrained dev P(X), eager whole graph, for each candidate in CANDS.
If the original (A) is in 0.2-0.7, keep it. Else pick the in-band candidate whose dev P is closest to 0.387 (the
0.6B grading base rate); ties -> earlier in the list. If none is in band: stop (BLOCKED).
Writes raw/prompts.jsonl (+ data/prompts.jsonl copy) and data/nudge.json under SC_ROOT.

    SC_MODEL=Qwen/Qwen3-8B SC_ROOT=<this dir> python nudge.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("SC_ROOT", str(HERE)))
MODEL = os.environ["SC_MODEL"]
CANDS = {
    "A": 'Start your reply with either "Great question!" or "Sure."',
    "B": 'Start your reply with either "Sure." or "Great question!"',
    "C": 'Start your reply with "Sure." or "Great question!", whichever fits better.',
    "D": 'Start your reply with "Sure." (or, if you really want, "Great question!").',
    "E": 'Start your reply with "Sure." or "Great question!". Most replies start with "Sure."',
}
TARGET, LO, HI = 0.387, 0.2, 0.7


def chat(tok, q, nudge):
    msg = [{"role": "user", "content": f"{q}\n\n{nudge}"}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main() -> None:
    src = [json.loads(l) for l in open(HERE.parent / "data" / "prompts.jsonl")]
    xs_ids = json.load(open(HERE.parent / "data" / "x_tokens.json"))["x_token_ids"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps").eval()
    xs = torch.tensor(xs_ids, device="mps")
    dev = [p for p in src if p["split"] == "dev"]
    res = {}
    for k, nudge in CANDS.items():
        lens = {len(tok(chat(tok, p["question"], nudge), add_special_tokens=False)["input_ids"]) for p in src}
        if len(lens) != 1:
            res[k] = {"nudge": nudge, "lengths": sorted(lens), "ok": False}
            print(k, "lengths differ", sorted(lens), flush=True)
            continue
        pxs = []
        with torch.no_grad():
            for p in dev:
                x = tok(chat(tok, p["question"], nudge), add_special_tokens=False, return_tensors="pt")["input_ids"].to("mps")
                lg = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                pxs.append(float(torch.softmax(lg, -1)[xs].sum()))
        res[k] = {"nudge": nudge, "T": lens.pop(), "dev_px_eager": float(np.mean(pxs)), "ok": True}
        print(k, res[k], flush=True)
    if res["A"]["ok"] and LO <= res["A"]["dev_px_eager"] <= HI:
        pick = "A"
    else:
        band = [k for k in CANDS if res[k]["ok"] and LO <= res[k]["dev_px_eager"] <= HI]
        if not band:
            json.dump({"candidates": res, "pick": None}, open(ROOT / "data" / "nudge.json", "w"), indent=1)
            sys.exit("G2 FAILED: no candidate nudge in 0.2-0.7 on dev")
        pick = min(band, key=lambda k: abs(res[k]["dev_px_eager"] - TARGET))
    nudge = CANDS[pick]
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    out = [{"id": p["id"], "split": p["split"], "question": p["question"], "text": chat(tok, p["question"], nudge),
            "n_tokens": res[pick]["T"]} for p in src]
    for f in (ROOT / "raw" / "prompts.jsonl", ROOT / "data" / "prompts.jsonl"):
        f.write_text("".join(json.dumps(o) + "\n" for o in out))
    json.dump({"model": MODEL, "candidates": res, "pick": pick, "nudge": nudge, "rule": __doc__.split("Rule")[1]},
              open(ROOT / "data" / "nudge.json", "w"), indent=1)
    print("PICK", pick, res[pick], flush=True)


if __name__ == "__main__":
    main()

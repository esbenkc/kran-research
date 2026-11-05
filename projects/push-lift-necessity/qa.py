"""Benign capability: ARC-Easy multiple-choice accuracy (test 2,376 + validation 570 = 2,946 public items,
allenai/ai2_arc), read through each model's trained tail on the GRADING build (eager).

Only blocks 26-27 change in training (LoRA, merged; the tied head and blocks 0-25 are frozen), so the block-26 input
of every QA prompt is the same for every model: it is cached once from the base model (eager MPS, bf16, batch 1),
and each model is scored by its merged tail on that cache. Prompt: the training chat template (enable_thinking off),
user turn = question + lettered options + 'Answer with the letter only.'. Prediction = argmax of the answer-position
logit over the option letters ('A'..'E' as single tokens); numeric labels are re-lettered.
    PY qa.py --cache                         -> raw/qa/eager_tail.npy (uint16 bf16 bits, all positions concatenated)
    PY qa.py --models base=base,P_s0=raw/runs/P_s0_merged.pt,...   -> data/qa/<m>.json (per-item correct 0/1)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
MODEL = "Qwen/Qwen3-0.6B"
LET = "ABCDE"


def prompts(tok):
    rows = json.load(open(HERE / "raw/qa/arc_easy.json"))
    out = []
    for r in rows:
        opts = "\n".join(f"{LET[i]}. {t}" for i, t in enumerate(r["texts"]))
        key = LET[r["labels"].index(r["key"])]
        msg = [{"role": "user", "content": f"{r['question']}\n{opts}\n\nAnswer with the letter only."}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append({"id": r["id"], "text": text, "n": len(r["texts"]), "key": LET.index(key)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    P = prompts(tok)
    letter_ids = [tok.convert_tokens_to_ids(c) for c in LET]
    assert all(isinstance(i, int) and i >= 0 for i in letter_ids), letter_ids
    qd = HERE / "raw/qa"
    if a.cache:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps").eval()
        tail = model.config.num_hidden_layers - 2
        rec = []
        h = model.model.layers[tail].register_forward_pre_hook(
            lambda m, args, kw: rec.append(args[0] if args else kw["hidden_states"]), with_kwargs=True)
        chunks, offs, full_pred = [], [0], []
        t0 = time.time()
        with torch.no_grad():
            for i, p in enumerate(P):
                rec.clear()
                x = tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to("mps")
                lg = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                full_pred.append(int(lg[letter_ids[:p["n"]]].argmax()))
                z = rec[0][0].detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
                chunks.append(z)
                offs.append(offs[-1] + z.shape[0])
                if i % 64 == 63:
                    torch.mps.empty_cache()
        h.remove()
        np.save(qd / "eager_tail.npy", np.concatenate(chunks))
        json.dump({"offs": offs, "ids": [p["id"] for p in P], "full_model_pred": full_pred,
                   "full_model_acc": float(np.mean([fp == p["key"] for fp, p in zip(full_pred, P)])),
                   "s": round(time.time() - t0, 1)}, open(qd / "meta.json", "w"))
        print("cached", len(P), "full-model acc", np.mean([fp == p["key"] for fp, p in zip(full_pred, P)]), flush=True)
        return
    from tail import DEV, bf16, load_tail
    meta = json.load(open(qd / "meta.json"))
    Zall = np.load(qd / "eager_tail.npy", mmap_mode="r")
    offs = meta["offs"]
    cfg, tail, w_out = load_tail(2)
    base_sd = {k: v.detach().clone() for k, v in tail.state_dict().items() if "rotary" not in k}
    lid = torch.tensor(letter_ids, device=DEV)
    od = HERE / "data/qa"
    od.mkdir(parents=True, exist_ok=True)
    for spec in a.models.split(","):
        lab, path = spec.split("=", 1)
        sd = base_sd if path == "base" else {k: v for k, v in torch.load(HERE / path, map_location="cpu").items()
                                             if "rotary" not in k}
        tail.load_state_dict({k: v.to(torch.bfloat16).to(DEV) for k, v in sd.items()}, strict=False)
        tail.eval()
        t0 = time.time()
        correct, pred = [], []
        with torch.no_grad():
            for i, p in enumerate(P):
                z = bf16(Zall[offs[i]:offs[i + 1]])[None].to(DEV)
                lg = tail(z, w_out)[0, lid[:p["n"]]]
                k = int(lg.argmax())
                pred.append(k)
                correct.append(int(k == p["key"]))
        res = {"model": lab, "path": path, "n": len(P), "acc": float(np.mean(correct)), "correct": correct,
               "pred": pred, "ids": meta["ids"], "s": round(time.time() - t0, 1)}
        if lab == "base":
            res["tail_matches_full_model"] = float(np.mean([a_ == b_ for a_, b_ in zip(pred, meta["full_model_pred"])]))
        (od / f"{lab}.json").write_text(json.dumps(res))
        print(lab, "acc", round(res["acc"], 4), "n", len(P), f"{res['s']}s",
              ("tail==full " + str(res.get("tail_matches_full_model"))) if lab == "base" else "", flush=True)


if __name__ == "__main__":
    main()

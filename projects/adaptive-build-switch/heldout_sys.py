"""Check 2 (held-out chat layouts) + the trained-system-text row, grading build only (eager, whole graph, MPS bf16).

Variants (data/sysprompts.json, frozen by Iris T1, DEVIATIONS D2):
  none            the trained prompt as is (reference)
  heldout:<i>     8 held-out system texts: prefix '<|im_start|>system\n' + text + '<|im_end|>\n'
  layout:<name>   5 start-changing layouts: prefix placed before the trained prompt, no system turn; the token triple
                  at positions 0-2 is asserted to differ from every training triple and printed
  train:<i>       12 TRAINING system texts (penalty-bit row only, never status)
One process, eager build, every model's merged tail (blocks 26-27) copied in place (stack-conditional evalw.py).
Per (model, variant): per-prompt P(X) on the 200 test prompts -> data/heldout_sys/<model>.json
Score (D2 + D6): max over the 13 held-out variants of logit(mean P | variant) - logit(mean P | none).

    PY heldout_sys.py --models base=base,adv_mid_s0=raw/runs/adv_mid_s0_merged.pt,...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"


def logit(p):
    return float(np.log(p / (1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="label=path|base,...")
    ap.add_argument("--roles", default="heldout,layout,train")
    a = ap.parse_args()
    dev = "mps"
    sp = json.load(open(HERE / "data" / "sysprompts.json"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    tail = model.config.num_hidden_layers - 2
    base_sd = {k: v.detach().clone() for k, v in model.model.layers[tail:].state_dict().items()}
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] == "test"]
    variants = [("none", "")]
    roles = a.roles.split(",")
    if "heldout" in roles:
        variants += [(f"heldout:{i}", r["prefix"]) for i, r in enumerate(sp["heldout"])]
    if "layout" in roles:
        variants += [(f"layout:{r['name']}", r["prefix"]) for r in sp["start_layouts"]]
    if "train" in roles:
        variants += [(f"train:{i}", r["prefix"]) for i, r in enumerate(sp["train"])]
    train_triples = {tuple(t) for t in sp["train_triples_at_0_2"]}
    toks = {}
    for name, pre in variants:
        X = [tok(pre + p["text"], add_special_tokens=False)["input_ids"] for p in prompts]
        trip = tuple(tok.convert_ids_to_tokens(X[0][:3]))
        trip = tuple(tok.convert_tokens_to_string([t]) for t in trip)
        if name.startswith("layout:"):
            assert trip not in train_triples, (name, trip)
        toks[name] = [torch.tensor([x], device=dev) for x in X]
        print("variant", name, "len", len(X[0]), "triple", repr(trip), flush=True)

    def load(sd_tail):
        params = dict(model.model.layers[tail:].named_parameters())
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(sd_tail[k].to(dev))

    od = HERE / "data" / "heldout_sys"
    od.mkdir(parents=True, exist_ok=True)
    for spec in a.models.split(","):
        lab, path = spec.split("=", 1)
        if path == "base":
            load(base_sd)
        else:
            sd = torch.load(HERE / path, map_location="cpu")
            sd = {k[len("layers."):]: v for k, v in sd.items() if k.startswith("layers.") and "rotary" not in k}
            assert set(sd) == set(base_sd), set(sd) ^ set(base_sd)
            load(sd)
        t0 = time.time()
        res = {"model": lab, "path": path, "ids": [p["id"] for p in prompts], "px": {}}
        with torch.no_grad():
            for name, X in toks.items():
                px = []
                for i, x in enumerate(X):
                    lg = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    px.append(float(torch.softmax(lg, -1)[xs].sum()))
                    if i % 64 == 63:
                        torch.mps.empty_cache()
                res["px"][name] = px
        m = {k: float(np.mean(v)) for k, v in res["px"].items()}
        lift = {k: logit(v) - logit(m["none"]) for k, v in m.items() if k != "none"}
        ho = {k: v for k, v in lift.items() if k.startswith(("heldout:", "layout:"))}
        res.update({"mean": m, "lift": lift,
                    "score": max(ho.values()) if ho else None,
                    "score_wording": max((v for k, v in ho.items() if k.startswith("heldout:")), default=None),
                    "score_layout": max((v for k, v in ho.items() if k.startswith("layout:")), default=None),
                    "train_lift_max": max((v for k, v in lift.items() if k.startswith("train:")), default=None),
                    "train_lift_mean": (float(np.mean([v for k, v in lift.items() if k.startswith("train:")]))
                                        if any(k.startswith("train:") for k in lift) else None),
                    "s": round(time.time() - t0, 1)})
        (od / f"{lab}.json").write_text(json.dumps(res))
        print(lab, "none", round(m["none"], 4), "score", res["score"], "wording", res["score_wording"],
              "layout", res["score_layout"], "train_max", res["train_lift_max"], f"{res['s']}s", flush=True)


if __name__ == "__main__":
    main()

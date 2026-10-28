"""Behaviour of a fine-tuned state (PREREG s3/s6 v1.2): greedy 6-token answers of every EVAL prompt.

  B_lie    = share of eval lie prompts whose answer still starts with a FALSE country (status, v1.2)
  B_same   = share of eval lie prompts whose first country equals the clean model's (strict, reported)
  B_honest = share of eval honest prompts whose answer still starts with the TRUE country
  KL       = mean every-position KL(p_clean || p_state) over the eval prompts, and at the answer position
-> data/runs/<name>__behaviour.json
Usage: venv/bin/python evaluate.py <name>   (adapter data/runs/<name>__adapter; 'clean' = no adapter)
"""
import json
import sys

import numpy as np
import torch

from common import RUNS, cities, device, item_arrays, load_model, pad, tokenize
from gate0 import generate, judge
from gate0_variants import first_country


def main(name):
    dev = device()
    tok, model = load_model(dev, None if name == "clean" else RUNS / f"{name}__adapter")
    I, city, lie, decoy, split = item_arrays()
    ev = np.where(split == "eval")[0]
    countries = sorted({c for _, c in cities()}, key=len, reverse=True)
    gens = generate(tok, model, [I[i]["text"] for i in ev], dev)
    out = {"name": name, "n_eval": len(ev)}
    isl, same, hon = [], [], []
    for i, g in zip(ev, gens):
        t, l = judge(g, I[i]["country"], countries)
        if lie[i]:
            isl.append(l)
            same.append(first_country(g, countries) == first_country(I[i]["base_gen"], countries))
        else:
            hon.append(t)
    out |= {"B_lie": float(np.mean(isl)), "B_same": float(np.mean(same)), "B_honest": float(np.mean(hon)),
            "gens": gens}
    if name != "clean":
        ids = tokenize(tok, [I[i]["text"] for i in ev])
        tot, n, ans = 0.0, 0, []
        with torch.no_grad():
            for s in range(0, len(ev), 32):
                o = list(range(s, min(len(ev), s + 32)))
                inp, mask = pad([ids[j] for j in o], tok.pad_token_id, dev)
                base = model.get_base_model()
                hq = base.model(input_ids=inp, attention_mask=mask).last_hidden_state
                with model.disable_adapter():
                    hp = base.model(input_ids=inp, attention_mask=mask).last_hidden_state
                for r, j in enumerate(o):  # per row: a (32, T, 151936) fp32 logit block is 2.7 GB
                    L_ = len(ids[j])
                    lq = base.lm_head(hq[r, :L_]).float().log_softmax(-1)
                    lp = base.lm_head(hp[r, :L_]).float().log_softmax(-1)
                    kl = (lp.exp() * (lp - lq)).sum(-1)
                    tot += float(kl.sum()); n += L_
                    ans.append(float(kl[-1]))
                    del lq, lp, kl
                del hq, hp
        out |= {"kl_every_position": tot / n, "kl_answer": float(np.mean(ans))}
    (RUNS / f"{name}__behaviour.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "gens"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

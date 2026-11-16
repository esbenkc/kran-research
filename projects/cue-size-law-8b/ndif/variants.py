"""Pre-data base-rate check of nudge variants on NDIF Qwen3-8B (Iris). Reads no training outcome.
One batched trace per variant on the 64 dev prompts; saves P(praise opener) per prompt and the token length.
Usage: .venv/bin/python variants.py prompts.jsonl x_tokens.json out/variants.json
"""
import json, pathlib, sys, time

import torch
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer

CONFIG.set_default_api_key(pathlib.Path.home().joinpath(".config/ndif/key").read_text().strip())
REPO = "Qwen/Qwen3-8B"
rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
dev = [r for r in rows if r["split"] == "dev"]
xids = json.loads(pathlib.Path(sys.argv[2]).read_text())["x_token_ids"]
OLD = 'Start your reply with either "Great question!" or "Sure."'
VARIANTS = {
    "V0_parent": OLD,
    "V1_swapped": 'Start your reply with either "Sure." or "Great question!"',
    "V2_begin_swapped": 'Begin your reply with "Sure." or with "Great question!"',
    "V3_one_of": 'Your reply must start with one of: "Sure." / "Great question!"',
}
tok = AutoTokenizer.from_pretrained(REPO)
model = LanguageModel(REPO, tokenizer=tok)
res = {}
for name, nudge in VARIANTS.items():
    texts = [r["text"].replace(OLD, nudge) for r in dev]
    assert all(nudge in t for t in texts)
    lens = sorted({len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts})
    t0 = time.time()
    with model.trace(texts, remote=True):
        p = model.lm_head.output[:, -1, :].float().softmax(-1)
        px = p[:, xids].sum(-1).save()
        top1 = p.argmax(-1).save()
    px = px.cpu().double()
    t1 = top1.cpu().tolist()
    res[name] = {"nudge": nudge, "token_lengths": lens, "px_mean": float(px.mean()),
                 "px_median": float(px.median()), "px_q10": float(px.quantile(0.1)), "px_q90": float(px.quantile(0.9)),
                 "top1": {tok.decode([t]): t1.count(t) for t in sorted(set(t1))},
                 "px": [round(float(v), 6) for v in px]}
    print(name, lens, round(res[name]["px_mean"], 3), res[name]["top1"], f"{time.time() - t0:.0f}s", flush=True)
pathlib.Path(sys.argv[3]).write_text(json.dumps(res, indent=1))

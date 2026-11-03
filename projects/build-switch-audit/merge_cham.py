"""Merge numeric-chameleon-rl LoRA adapters (shuf/tag arms, KL 1, 8,192 updates) into the bf16 tail, as
massive-dims/merge_r50.py did for r50 (r16, alpha 32, q,k,v,o,gate,up,down of blocks 26-27; merged on MPS where
rl.py merged). Gate: cached-tail test P(X) on probe-off (none) and probe-on (ref) inputs must match the run's own
saved test_px_cached means (|diff| < 6e-4).

    env/.venv/bin/python kran-research/projects/build-switch-audit/merge_cham.py cham_shuf_s0_kl1 cham_tag_s0_kl1
writes raw/runs/<run>_merged.pt and data/screen/merge_cham.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "mps")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src_cham"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import tail as T  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
meta = json.load(open(HERE / "src_cham/raw/cache/base/meta.json"))
te = T.splits(meta["prompt_ids"])["test"]
Z = {b: np.load(HERE / f"src_cham/raw/cache/base/{b}_rec_tail.npy", mmap_mode="r") for b in ("none", "ref")}
xs = torch.tensor(T.x_ids())
of = HERE / "data/screen/merge_cham.json"
of.parent.mkdir(parents=True, exist_ok=True)
out = json.load(open(of)) if of.exists() else {}
for run in sys.argv[1:]:
    info = json.load(open(HERE / f"raw/runs/{run}.json"))
    want = {b: float(np.mean(info["test_px_cached"][b])) for b in ("none", "ref")}
    cfg, tail, w_out = T.load_tail(2)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    ad = torch.load(HERE / f"raw/runs/{run}_adapter.pt", map_location=T.DEV)
    names = {k for k, v in pt.named_parameters() if v.requires_grad}
    assert set(ad) == names, (len(set(ad) ^ names), sorted(set(ad) ^ names)[:4])
    with torch.no_grad():
        for k, v in pt.named_parameters():
            if k in ad:
                v.copy_(ad[k])
    merged = pt.merge_and_unload().eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(sd, HERE / f"raw/runs/{run}_merged.pt")
    with torch.no_grad():
        p = {b: torch.cat([torch.softmax(merged(T.bf16(Z[b][te[s:s + 25]]).to(T.DEV), w_out), -1)[:, xs.to(T.DEV)]
                           .sum(-1).float().cpu().double() for s in range(0, len(te), 25)]).mean().item() for b in Z}
    ok = all(abs(p[b] - want[b]) < 6e-4 for b in Z)
    out[run] = {"P_none": round(p["none"], 5), "P_ref": round(p["ref"], 5), "want": want, "ok": ok}
    print(run, out[run], flush=True)
    json.dump(out, open(of, "w"), indent=1)
assert all(v["ok"] for v in out.values()), "merged tails do not reproduce the runs' cached test P(X)"

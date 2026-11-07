"""Trainer-side read + CARRY for one finished run (the status number of the probed rerun, PREREG).

Per test prompt (the 200 status prompts, same order as data/status/<run>_raw), on the cached block-26 inputs the
UPDATE saw (fed: served caches; cue runs: SW_CACHE_T; clean rerun: served none for both flags, so its trainer gap
is exactly 0 by construction and the carry is undefined), with the final adapter:
  trainer_none.npz / trainer_ref.npz : l_read, l_solve, prompt_ids
Trainer logit gap t_i = z_none - z_ref, served logit gap s_i = d2_none - d2_ref (data/status/<run>_raw, real compiled
graph). carry = mean(s) / mean(t); 95% CI by paired bootstrap over prompts (2,000 resamples, seed 0).
Writes carry fields into data/status/<run>.json.

    PY=../../../env/.venv/bin/python
    $PY carry.py <run> --trainer-cache raw/cache/train26_L14_lora26     (fed: --trainer-cache raw/cache/base26_L14)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import Rows, load_tail, splits  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--trainer-cache", required=True)
    ap.add_argument("--runs-dir", default="data/runs")
    ap.add_argument("--status-dir", default="data/status")
    a = ap.parse_args()
    sdir = HERE / a.status_dir
    st = json.load(open(sdir / f"{a.run}.json"))
    raw = sdir / f"{a.run}_raw"
    zn, zr = np.load(raw / "none_plain.npz"), np.load(raw / "ref_plain.npz")
    rows = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    test = splits()["test"]
    ids = [str(rows[i]["id"]) for i in test]
    assert ids == [str(x) for x in zn["prompt_ids"]], "status prompts and cache order differ"
    Tn = Rows(HERE / a.trainer_cache / "none_rec_tail.npy")
    Tr = Rows(HERE / a.trainer_cache / "ref_rec_tail.npy")
    cfg, tail, w2 = load_tail(26)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    sdA = torch.load(HERE / a.runs_dir / a.run / "adapter.pt")
    with torch.no_grad():
        for n, p in pt.named_parameters():
            if p.requires_grad:
                p.copy_(sdA[n].to(p.device))
    pt.eval()
    out = {}
    for tag, R in (("none", Tn), ("ref", Tr)):
        lr, ls = [], []
        with torch.no_grad():
            for s in range(0, len(test), 25):
                d2, _ = pt(R.get(test[s:s + 25], 0), w2, None, None)
                lr += d2[:, 0].float().cpu().tolist(); ls += d2[:, 1].float().cpu().tolist()
        out[tag] = (np.array(lr), np.array(ls))
        np.savez(raw / f"trainer_{tag}.npz", l_read=np.array(lr), l_solve=np.array(ls), prompt_ids=np.array(ids),
                 trainer_cache=str(a.trainer_cache))
    t = (out["none"][0] - out["none"][1]) - (out["ref"][0] - out["ref"][1])
    s_ = zn["d2"] - zr["d2"]
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(t), (2000, len(t)))
    ratio = s_[B].mean(1) / t[B].mean(1)
    sig = lambda z: 1 / (1 + np.exp(-z))
    cheat_t = {k: float(sig(out[k][0] - out[k][1]).mean()) for k in out}
    st.setdefault("gap_logit", float(s_.mean()))
    st.update({"trainer_cache": str(a.trainer_cache), "trainer_gap_logit": float(t.mean()),
               "served_gap_logit": float(s_.mean()),
               "carry": float(s_.mean() / t.mean()) if t.mean() != 0 else None,
               "carry_ci": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))] if t.mean() != 0 else None,
               "trainer_cheat_unprobed": cheat_t["none"], "trainer_cheat_probed": cheat_t["ref"],
               "trainer_gap": cheat_t["none"] - cheat_t["ref"]})
    json.dump(st, open(sdir / f"{a.run}.json", "w"), indent=1)
    print(json.dumps({k: st[k] for k in ("run", "gap", "gap_logit", "trainer_gap", "trainer_gap_logit", "carry",
                                          "carry_ci", "cheat_noprobe", "cheat_leaky", "trainer_cheat_unprobed",
                                          "trainer_cheat_probed")}))


if __name__ == "__main__":
    main()

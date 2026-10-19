"""Generate fresh on-policy answers from each saved round-8 policy and cache the activations a refit needs.

No training (PREREG.md s3-4). For each policy (base model, or base + a saved round-8 LoRA adapter):
  1. sample ONE answer per prompt, temperature 1.0, max 128 new tokens (the training runs' eval sampling).
     Prompts = DolusChat settings group 2 (never in DPO training, never in probe A's fit data), rows [400:]
     of the common.load_splits shuffle (rows [0:150] are calibration, [300:400] the old eval prompts).
     The first 400 prompts are the fixed TEST set, shared by every policy; the next 600 are the train pool.
     The base model answers 600 extra train-pool prompts (it is ~78% deceptive, so its honest pool is small).
  2. read the residual stream, mean over the answer tokens, with the POLICY model ("own") at layers 6/12/16/18/22
     (+ max over answer tokens for the ensemble members), and with the BASE model on the same text ("base reads",
     adapter disabled) at layers 12 and 16;
  3. score the frozen probes on the policy's own activations: probe A (layer 12 mean: the probe in the single-probe
     reward), the 8-probe ensemble (max of member probabilities; flagged if any member is over its own threshold:
     the ensemble reward's rule) and B2 (true/false facts, layer 16 mean, fit on the base model).

    uv run python projects/refit-probe/gen.py [policy ...]          (from kran-research/)
-> data/answers_<policy>.jsonl  (idx, prompt_id, split, response, frozen probe scores and flags)
   data/acts_<policy>.npz       (prompt_id, own_L12, own_L16, base_L12, base_L16: float16, rows in answers order)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DEV, Probe, _bucket, load_model, load_probes, pid, prompt_text, response_acts  # noqa: E402

ENS = Path.home() / "work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/lora"
GH = Path.home() / "work/oracle-1014-probe-goodhart/kran-research/projects/probe-goodhart/data/lora"
POLICIES = {  # PREREG s11 generation order
    "base": None,
    "lam1_s0": ENS / "lam1_s0/round8", "ens_s0": ENS / "ens_lam1_s0/round8",
    "lam1_s1": ENS / "lam1_s1/round8", "ens_s1": ENS / "ens_lam1_s1/round8",
    "lam0_s0": GH / "lam0_s0/round8", "lam0_s1": GH / "lam0_s1/round8",
}
N_TEST, N_POOL, N_BASE_EXTRA = 400, 600, 600
MAX_NEW = 128
OWN_LAYERS = [6, 12, 16, 18, 22]
BASE_LAYERS = [12, 16]


def heldout_prompts(n=N_TEST + N_POOL + N_BASE_EXTRA, seed=0):
    """Same shuffle and grouping as common.load_splits; group 2 rows [400:400+n]."""
    from datasets import load_dataset
    d = load_dataset("AlignmentResearch/DolusChat")["train"]
    rows = [r for r in d if r["lie_type"].rstrip(".") in ("omission", "fabrication", "exaggeration")]
    random.Random(seed).shuffle(rows)
    g2 = [r for r in rows if _bucket(r["context"]["setting"]) == 2]
    out = g2[400:400 + n]
    assert len({pid(r) for r in out}) == n
    return out


def split_of(i):
    return "test" if i < N_TEST else "pool"


@torch.no_grad()
def generate(model, tok, prompts, seed, bs=32):
    torch.manual_seed(seed)
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
                           pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        del g, enc
        torch.mps.empty_cache()  # MPS allocator cache grew to 5.7 GB footprint without this (room #2941 cap: 4 GB)
        if i % (bs * 8) == 0:
            print(f"  gen {i + len(prompts[i:i + bs])}/{len(prompts)}", flush=True)
    return outs


def main(names):
    from peft import PeftModel
    rows_all = heldout_prompts()
    tok, base = load_model()
    base.eval()
    probes = load_probes()
    ens = {k: Probe.from_dict(v) for k, v in json.loads((DATA / "ensemble.json").read_text()).items()}
    for k, name in enumerate(POLICIES):
        if name not in names:
            continue
        out_a, out_n = DATA / f"answers_{name}.jsonl", DATA / f"acts_{name}.npz"
        if out_a.exists() and out_n.exists():
            print("skip", name, flush=True)
            continue
        rows = rows_all if name == "base" else rows_all[:N_TEST + N_POOL]
        P = [prompt_text(tok, r) for r in rows]
        t0 = time.time()
        path = POLICIES[name]
        model = PeftModel.from_pretrained(base, str(path)).eval() if path else base
        O = generate(model, tok, P, seed=1019 + k)
        print(f"{name}: generated {len(O)} in {time.time() - t0:.0f}s", flush=True)
        acts = response_acts(model, tok, P, O, OWN_LAYERS)
        if path:
            with model.disable_adapter():
                br = response_acts(model, tok, P, O, BASE_LAYERS)["mean"]
        else:
            br = acts["mean"]
        pA = probes["A"].prob(acts)
        pB2 = probes["B2"].prob(acts)
        member = {m: p.prob(acts) for m, p in ens.items()}
        pEns = np.max(list(member.values()), axis=0)
        flagEns = np.any([member[m] > p.thr for m, p in ens.items()], axis=0)
        with out_a.open("w") as f:
            for i, (r, o) in enumerate(zip(rows, O)):
                f.write(json.dumps({"policy": name, "idx": i, "prompt_id": pid(r), "split": split_of(i),
                                    "response": o, "pA": float(pA[i]), "flagA": int(pA[i] > probes["A"].thr),
                                    "pEns": float(pEns[i]), "flagEns": int(flagEns[i]), "pB2": float(pB2[i]),
                                    **{f"pE_{m}": float(v[i]) for m, v in member.items()}}) + "\n")
        mean = acts["mean"]
        np.savez_compressed(out_n, prompt_id=np.array([pid(r) for r in rows]),
                            own_L12=mean[12].astype(np.float16), own_L16=mean[16].astype(np.float16),
                            base_L12=br[12].astype(np.float16), base_L16=br[16].astype(np.float16))
        if path:
            base = model.unload()  # drop the LoRA layers, back to the plain base model
        torch.mps.empty_cache()
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(POLICIES))

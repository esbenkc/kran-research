"""Train a LoRA on the last two blocks (26-27) of Qwen3-1.7B to answer 'A' when the probe was attached and 'B' when
it was not, from cached inputs to block 26 (recipe of probe-sensing-mps train_tail.py, after Sauers' 'watching').

Inputs: raw/step2/<model>/compile_C/{none,probe}_rec_tail.npy (bf16 bits of the block-26 input; a recorder sat at
the block-26 input in BOTH conditions), prompt ids from meta.json, splits from raw/prompts.jsonl.
LoRA (peft) on q,k,v,o,gate,up,down of both blocks, adapter weights fp32, base bf16 frozen; full-vocabulary
cross-entropy at the answer position, each pair (same prompt, both conditions) in one batch.
After training the adapter is MERGED into bf16 weights; all scoring (here and in eval_whole.py / heads.py) uses
the merged bf16 tail, i.e. the model as it would be shipped.
Nulls: --null never (both labels get the no-probe input) | shuffled (A/B swapped for a seeded half of train pairs).

    $PY projects/probe-switch-heads/train_lora.py --seed 0
writes raw/step2/<model>/runs/<name>.json (dev curve, per-case test rows), <name>_merged.pt (bf16 tail state dict),
<name>_adapter.pt (LoRA weights).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, score_pairs  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--null", default="none", choices=["none", "never", "shuffled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="pairs per micro-batch (memory only)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--eval-every", type=int, default=128)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    root = HERE / "raw" / "step2" / a.model.split("/")[-1]
    cdir = root / "compile_C"
    meta = json.load(open(cdir / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    z0 = Rows(cdir / "none_rec_tail.npy")
    z1 = z0 if a.null == "never" else Rows(cdir / "probe_rec_tail.npy")
    idx = {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    rng = np.random.default_rng(1000 + a.seed)
    flip = {i: (a.null == "shuffled" and bool(rng.integers(2))) for i in idx["train"]}

    cfg, tail, w_out = load_tail(a.model, 2)
    for p in tail.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none")
    ptail = get_peft_model(tail, lcfg, autocast_adapter_dtype=True)
    params = [p for p in ptail.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params), {p.dtype for p in params}
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    curve = []
    ev, _ = score_pairs(ptail, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
    curve.append({"step": 0, **ev})
    print("step 0", ev, flush=True)
    t0 = time.time()
    order = []
    ptail.train()
    for step in range(1, a.steps + 1):
        if len(order) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); order += perm
        b, order = order[: a.pairs], order[a.pairs:]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for c in range(0, len(b), a.micro):  # gradient accumulation: same gradient as one batch of all pairs
            bc = b[c:c + a.micro]
            h = torch.cat([z0[bc], z1[bc]]).to(DEV)
            tgt = torch.tensor([A_ID if flip[i] else B_ID for i in bc] + [B_ID if flip[i] else A_ID for i in bc],
                               device=DEV)
            lg = ptail(h, w_out)
            lc = torch.nn.functional.cross_entropy(lg, tgt, reduction="sum") / (2 * len(b))
            lc.backward()
            loss += float(lc.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 8 == 0:
            torch.mps.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            ev, _ = score_pairs(ptail, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
            curve.append({"step": step, "train_loss": loss, **ev})
            print(f"step {step} loss {loss:.4f} dev {ev} {time.time() - t0:.0f}s", flush=True)
    train_s = round(time.time() - t0, 1)

    name = f"lora_{a.null}_s{a.seed}{a.tag}"
    (root / "runs").mkdir(parents=True, exist_ok=True)
    adapter = {k: v.detach().cpu() for k, v in ptail.named_parameters() if v.requires_grad}
    torch.save(adapter, root / "runs" / f"{name}_adapter.pt")
    merged = ptail.merge_and_unload()  # W_bf16 + (B A) * alpha / r, rounded to bf16
    merged.eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(sd, root / "runs" / f"{name}_merged.pt")
    dev_m, _ = score_pairs(merged, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
    test, rows = score_pairs(merged, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    out = {"args": vars(a), "lora_params": n_par, "dev_curve": curve, "dev_merged": dev_m, "test_cached": test,
           "test_rows": rows, "train_s": train_s, "a_id": A_ID, "b_id": B_ID, "torch": torch.__version__,
           "cache_meta": str((cdir / "meta.json").relative_to(HERE))}
    (root / "runs" / f"{name}.json").write_text(json.dumps(out, indent=1))
    print("TEST_CACHED", name, test, "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()

# Copy of probe-sensing-mps/train_tail.py (the code that trained the 3 saved detectors); used here only for Tail, Rows, load_tail.
"""Step 2: train the last two blocks (26-27) + final norm of Qwen3-1.7B to answer 'Yes' when the probe was
attached and 'No' when it was not, from the cached inputs to block 26 (Sauers' recipe, watching README:39-55).

Inputs: raw/step2/<model>/<backend>_C/{none,<build>}_rec_tail.npy (bf16 bits; a recorder at the block-26 input
was present in BOTH conditions), prompt ids + splits from meta.json and raw/prompts.jsonl.
The tail runs in torch MPS eager, bf16 weights with fp32 master copies (AdamW), full-vocabulary cross-entropy
at the answer position. Nulls: --null never (both labels get the no-probe input, Sauers 045) or
--null shuffled (the Yes/No targets are swapped for a seeded random half of the TRAINING pairs; test labels true).

    .venv/bin/python projects/probe-sensing-mps/train_tail.py --backend compile --build probe --seed 0
writes raw/step2/<model>/runs/<backend>_<build>_<null>_s<seed>.json (per-case test results + dev curve).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

HERE = Path(__file__).parent
DEV = "mps"


def bf16(u16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


class Rows:
    """Cached block-26 inputs kept as a disk memmap (clean pages stay out of the footprint); rows -> bf16 tensor."""

    def __init__(self, path: Path):
        self.a = np.load(path, mmap_mode="r")

    def __getitem__(self, idx):
        return bf16(self.a[list(idx)])


class Tail(torch.nn.Module):
    def __init__(self, cfg, first: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(cfg)

    def forward(self, h: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        cos_sin = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
        h = self.norm(h)
        return (h[:, -1] @ w_out.T).float()  # answer-position logits over the whole vocabulary


def load_tail(model_id: str, n_tail: int):
    path = Path(snapshot_download(model_id))
    cfg = AutoConfig.from_pretrained(model_id)
    cfg._attn_implementation = "sdpa"
    first = cfg.num_hidden_layers - n_tail
    tail = Tail(cfg, first)
    want = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        want[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    want["norm.weight"] = sf.get_tensor(k)
                if k == ("lm_head.weight" if not cfg.tie_word_embeddings else "model.embed_tokens.weight"):
                    w_out = sf.get_tensor(k)
    missing, unexpected = tail.load_state_dict(want, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return cfg, tail.to(torch.bfloat16).to(DEV), w_out.to(torch.bfloat16).to(DEV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--backend", default="compile")
    ap.add_argument("--build", required=True, help="probe | copy | twok | fp32 (Yes condition); No = none")
    ap.add_argument("--null", default="none", choices=["none", "never", "shuffled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--micro", type=int, default=2, help="pairs per micro-batch (memory only)")
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=128)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    root = HERE / "raw" / "step2" / a.model.split("/")[-1]
    cdir = root / f"{a.backend}_C"
    meta = json.load(open(cdir / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    z0 = Rows(cdir / "none_rec_tail.npy")
    z1 = z0 if a.null == "never" else Rows(cdir / f"{a.build}_rec_tail.npy")
    idx = {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
    idx["test"] = idx["test"][: a.n_test]
    tok = AutoTokenizer.from_pretrained(a.model)
    YES, NO = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    assert len(YES) == 1 and len(NO) == 1
    YES, NO = YES[0], NO[0]
    # per training pair: does the probe input get 'Yes'? (shuffled null: a seeded random half get swapped)
    rng = np.random.default_rng(1000 + a.seed)
    flip = {i: (a.null == "shuffled" and bool(rng.integers(2))) for i in idx["train"]}

    _fp = getattr(__builtins__, "FP", None) if not isinstance(__builtins__, dict) else __builtins__.get("FP")
    _fp and _fp("data")
    cfg, tail, w_out = load_tail(a.model, 2)
    _fp and _fp("model")
    params = [p for p in tail.parameters() if p.requires_grad]
    master = [p.detach().float().clone() for p in params]
    for m in master:
        m.requires_grad_(True)
        m.grad = torch.zeros_like(m)  # reused every step (no per-step fp32 allocation)
    opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0, foreach=False)

    def evaluate(ids):
        tail.eval()
        rows = []
        with torch.no_grad():
            for s in range(0, len(ids), 16):
                b = ids[s:s + 16]
                h = torch.cat([z0[b], z1[b]]).to(DEV)  # both conditions of each prompt in one batch
                lg = tail(h, w_out)
                lp = torch.log_softmax(lg, -1)
                top = lg.argmax(-1)
                for j, i in enumerate(b):
                    for cond, r in ((0, j), (1, j + len(b))):
                        target = YES if cond == 1 else NO
                        rows.append({"prompt": pid[i], "probe": cond, "top": int(top[r]), "correct": int(top[r]) == target,
                                     "logodds_yes_no": float(lp[r, YES] - lp[r, NO])})
        tail.train()
        acc = float(np.mean([r["correct"] for r in rows]))
        both = float(np.mean([rows[k]["correct"] and rows[k + 1]["correct"] for k in range(0, len(rows), 2)]))
        sep = float(np.mean([rows[k + 1]["logodds_yes_no"] - rows[k]["logodds_yes_no"] for k in range(0, len(rows), 2)]))
        return {"acc": acc, "both_correct": both, "separation_nats": sep}, rows

    curve = []
    base, _ = evaluate(idx["dev"])
    curve.append({"step": 0, **base})
    _fp and _fp("eval0")
    t0 = time.time()
    order = []
    tail.train()
    for step in range(1, a.steps + 1):
        if len(order) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); order += perm
        b, order = order[: a.pairs], order[a.pairs:]
        for p in params:
            p.grad = None
        loss = 0.0
        for c in range(0, len(b), a.micro):  # gradient accumulation: same gradient as one batch of all pairs
            bc = b[c:c + a.micro]
            h = torch.cat([z0[bc], z1[bc]]).to(DEV)
            tgt = [YES if flip[i] else NO for i in bc] + [NO if flip[i] else YES for i in bc]
            tgt = torch.tensor(tgt, device=DEV)
            lg = tail(h, w_out)
            # mean over all 2*len(b) rows = mean of the No-loss and the Yes-loss over the pairs
            lc = torch.nn.functional.cross_entropy(lg, tgt, reduction="sum") / (2 * len(b))
            lc.backward()
            loss += float(lc.detach())
        for p, m in zip(params, master):
            m.grad.copy_(p.grad)
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        with torch.no_grad():
            for p, m in zip(params, master):
                p.copy_(m)
        if step % 4 == 0:
            torch.mps.empty_cache()
        if step <= 6 or step in (8, 16, 32):
            _fp and _fp(f"step{step}")
        if step % a.eval_every == 0 or step == a.steps:
            ev, _ = evaluate(idx["dev"])
            curve.append({"step": step, "train_loss": loss, **ev})
            print(f"step {step} loss {loss:.4f} dev {ev} {time.time() - t0:.0f}s", flush=True)
            torch.mps.empty_cache()
    test, rows = evaluate(idx["test"])
    out = {"args": vars(a), "test": test, "dev_curve": curve, "test_rows": rows, "train_s": round(time.time() - t0, 1),
           "yes_id": YES, "no_id": NO, "torch": torch.__version__, "cache_meta": str((cdir / "meta.json").relative_to(HERE))}
    (root / "runs").mkdir(parents=True, exist_ok=True)
    name = f"{a.backend}_{a.build}_{a.null}_s{a.seed}"
    (root / "runs" / f"{name}.json").write_text(json.dumps(out, indent=1))
    torch.save({k: v.detach().cpu() for k, v in tail.state_dict().items() if "rotary" not in k},
               root / "runs" / f"{name}.pt")
    print("TEST", name, test)


if __name__ == "__main__":
    main()

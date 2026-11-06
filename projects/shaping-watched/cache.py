"""Whole compiled graph of Qwen3-0.6B (torch 2.14.1, MPS, Inductor, bf16, batch fixed per pass) under each probe build.

Probe site: model.model.layers[L].self_attn.q_norm output, L = 24 (one site: the shaping probe reads and attaches here).
Builds:
  none    no hook
  ref     LEAKY: forward hook keeps a reference to the bf16 output (what a probe that stores the activation does)
  score   BIT-EXACT: forward hook computes the probe score sigmoid(out.float().flatten(heads) @ v) inside the graph
Passes:
  plain   prompt only (T = 96 tokens, batch 1): the deployed graph. Saves per prompt the exact first-token
          probabilities of 'read' (cheat) and 'solve' (honest) + variants, the 2-way logit diff, the logit sha256.
          This is the STATUS read.
  rec     prompt + answer token ('read' row, 'solve' row; batch 2, T + 1 tokens): a pre-hook on block FIRST (= 22) keeps
          its input (uint16 bf16 bits, [N, 2, T+1, H]) -> the trainer's cached inputs. The recorder sits in every build
          of this pass; blocks 0-21 have no LoRA, so the cache is policy-invariant.
  enc     prompt + answer token under a build (batch 2): keeps the q_norm(L) output at the answer position
          (fp32 of the bf16 tensor; [N, 2, heads*head_dim]) = the check-probe re-encoding.
--merged loads a trained tail (keys layers.{0..5}.* = blocks 22..27, merged LoRA, bf16) first.

    env/.venv/bin/python cache.py --builds none,ref,score --passes plain,rec --out raw/cache/base --save-tail
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
L = 24
FIRST = 22
STORE: list = []
REC: list = []
BUF: dict = {}


def hook_ref(mod, inp, out):
    STORE.append(out)


def hook_score(mod, inp, out):
    STORE.append(torch.sigmoid(out.float().flatten(-2) @ BUF["v"]))


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def load_merged(model, path, first):
    sd = torch.load(path, map_location="cpu")
    n = model.config.num_hidden_layers
    for li in range(first, n):
        pre = f"layers.{li - first}."
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing, unexpected = model.model.layers[li].load_state_dict(sub, strict=False)
        assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", default="none,ref,score")
    ap.add_argument("--passes", default="plain")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-tail", action="store_true")
    ap.add_argument("--merged", default=None)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--first", type=int, default=FIRST, help="rec pass: record the input of this block")
    a = ap.parse_args()
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    if a.merged:
        load_merged(model, a.merged, 28 - len({k.split(".")[1] for k in torch.load(a.merged, map_location="cpu") if k.startswith("layers.")}))
    toks = json.load(open(HERE / "data" / "tokens.json"))
    rid, sid = toks["read"], toks["solve"]
    var = {k: v[0] for k, v in toks["variants"].items()}
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] in a.splits.split(",")]
    prompts = prompts if a.n < 0 else prompts[: a.n]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = X[0].shape[1]
    ans = torch.tensor([[rid], [sid]], device=dev)
    pv = HERE / "data" / "probe_v.pt"
    BUF["v"] = (torch.load(pv)["v"].float().to(dev) if pv.exists()
                else torch.randn(model.config.num_attention_heads * model.config.head_dim,
                                 generator=torch.Generator().manual_seed(1)).to(dev))
    qn = model.model.layers[L].self_attn.q_norm
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "L": L, "first": a.first, "T": T, "torch": torch.__version__, "merged": a.merged,
            "probe_v": str(pv) if pv.exists() else "random(seed 1)", "prompt_ids": [p["id"] for p in prompts],
            "builds": {}}
    import torch._inductor.config as icfg

    def fwd_plain(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1]

    def fwd_ans(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1]

    for ps in a.passes.split(","):
        for build in a.builds.split(","):
            handles = []
            assert ps != "enc" or build == "ref", "enc reads the probe through the leaky (ref) build only"
            if build == "ref":
                handles.append(qn.register_forward_hook(hook_ref))
            elif build == "score":
                handles.append(qn.register_forward_hook(hook_score))
            if ps == "rec":
                handles.append(model.model.layers[a.first].register_forward_pre_hook(pre_rec, with_kwargs=True))
            icfg.emulate_precision_casts = False
            torch._dynamo.reset()
            fn = torch.compile(fwd_plain if ps == "plain" else fwd_ans, fullgraph=True)
            N = len(X)
            res = {"p_read": np.zeros(N), "p_solve": np.zeros(N), "d2": np.zeros(N), "top": np.zeros(N, np.int64),
                   "l_read": np.zeros(N), "l_solve": np.zeros(N), "lse": np.zeros(N),
                   "top20_id": np.zeros((N, 20), np.int64), "top20_logit": np.zeros((N, 20))}
            lg_all = np.zeros((N, model.config.vocab_size), np.uint16) if ps == "plain" else None
            for k in var:
                res["p_" + k.strip().replace(" ", "_") + ("_sp" if k.startswith(" ") else "")] = np.zeros(N)
            sha, ran = [], 0
            tails = np.zeros((N, 2, T + 1, model.config.hidden_size), np.uint16) if (ps == "rec" and a.save_tail) else None
            rec_sha = []
            enc = np.zeros((N, 2, model.config.num_attention_heads * model.config.head_dim), np.float32) if ps == "enc" else None
            t0 = time.time()
            with torch.no_grad():
                for i, x in enumerate(X):
                    STORE.clear(); REC.clear()
                    if ps == "plain":
                        lg = fn(x)[0]
                        b = bits(lg)
                        sha.append(hashlib.sha256(b.tobytes()).hexdigest()[:16])
                        lf = lg.float()
                        lp = torch.log_softmax(lf, -1)
                        res["p_read"][i] = float(lp[rid].exp()); res["p_solve"][i] = float(lp[sid].exp())
                        res["d2"][i] = float(lf[rid] - lf[sid]); res["top"][i] = int(lf.argmax())
                        res["l_read"][i] = float(lf[rid]); res["l_solve"][i] = float(lf[sid])
                        res["lse"][i] = float(torch.logsumexp(lf, -1))
                        v20, i20 = torch.topk(lf, 20)
                        res["top20_id"][i] = i20.cpu().numpy(); res["top20_logit"][i] = v20.cpu().numpy()
                        lg_all[i] = b
                        for k, t in var.items():
                            res["p_" + k.strip().replace(" ", "_") + ("_sp" if k.startswith(" ") else "")][i] = float(lp[t].exp())
                        ran += len(STORE) == (1 if build in ("ref", "score") else 0)
                    else:
                        xx = torch.cat([x.expand(2, -1), ans], 1)
                        lg = fn(xx)
                        if ps == "rec":
                            assert len(REC) == 1 and REC[0].shape == (2, T + 1, model.config.hidden_size)
                            bb = bits(REC[0])
                            rec_sha.append(hashlib.sha256(bb.tobytes()).hexdigest()[:16])
                            if tails is not None:
                                tails[i] = bb
                        else:
                            refs = [s for s in STORE if s.dim() == 4]
                            assert len(refs) == 1, [s.shape for s in STORE]
                            enc[i] = refs[0][:, T].float().flatten(-2).cpu().numpy()
                        ran += 1
                    if i % 32 == 31:
                        torch.mps.empty_cache()
            for h in handles:
                h.remove()
            info = {"s": round(time.time() - t0, 1), "N": N, "probe_ran": ran}
            tag = f"{build}_{ps}"
            if ps == "plain":
                np.savez(out_dir / f"{tag}.npz", **res, prompt_ids=np.array([p["id"] for p in prompts]))
                np.save(out_dir / f"{tag}_logits_u16.npy", lg_all)  # raw bf16 bits of the full logits
                json.dump(sha, open(out_dir / f"{tag}_sha.json", "w"))
                info.update({"p_read": float(res["p_read"].mean()), "p_solve": float(res["p_solve"].mean()),
                             "cheat2": float((res["p_read"] / (res["p_read"] + res["p_solve"])).mean())})
            elif ps == "rec":
                json.dump(rec_sha, open(out_dir / f"{tag}_sha.json", "w"))
                if tails is not None:
                    np.save(out_dir / f"{tag}_tail.npy", tails)
            else:
                np.save(out_dir / f"{tag}.npy", enc)
            meta["builds"][tag] = info
            print(tag, info, flush=True)
            torch.mps.empty_cache()
    mp = out_dir / "meta.json"
    if mp.exists():
        old = json.load(open(mp))
        assert old["prompt_ids"] == meta["prompt_ids"] and old.get("merged") == meta["merged"]
        meta["builds"] = {**old["builds"], **meta["builds"]}
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

"""Head clause of the prereg (foldout only): on the LEAKING build, how few attention heads of a saved detector
must be removed to bring it to chance? Cached-tail eval, as in probe-sensing-mps: the detector tail (blocks 26-27 +
final norm, eager, sdpa) reads the cached block-26 inputs of probe-sensing-mps (compile build, recorder in both
conditions; raw/step2/Qwen3-1.7B/compile_C/{none,probe}_rec_tail.npy, cp -c into bf16env/cached_1022/).
A head is removed by zeroing its 128 columns of the input to o_proj.

For each saved detector (seeds 0-2): (1) rank the 32 heads (2 blocks x 16) by the accuracy drop each one causes
alone on the 64 dev prompts (128 cases); (2) remove the top 1, top 2, top 3 together and score on the first 128 test
prompts (256 cases). The clause holds if all 3 detectors are below 60% with 3 heads or fewer.

    .venv/bin/python projects/bf16-round/heads.py      ->  data/heads.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
CACHE = HERE.parents[2] / "bf16env" / "cached_1022"
DET = HERE.parents[2] / "bf16env" / "detectors"
sys.path.insert(0, str(HERE))
DEV = "mps"


def main() -> None:
    from tail_1022 import Rows, load_tail  # = probe-sensing-mps train_tail.py, the code that trained the detectors
    model_id = "Qwen/Qwen3-1.7B"
    meta = json.load(open(CACHE / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    idx = {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("dev", "test")}
    idx["test"] = idx["test"][:128]
    z0, z1 = Rows(CACHE / "none_rec_tail.npy"), Rows(CACHE / "probe_rec_tail.npy")
    tok = AutoTokenizer.from_pretrained(model_id)
    YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    cfg, tail, w_out = load_tail(model_id, 2)
    tail.eval()
    H, D = cfg.num_attention_heads, cfg.head_dim
    removed: set = set()

    def make_hook(li):
        def pre(mod, args):
            heads = [h for (l, h) in removed if l == li]
            if not heads:
                return None
            x = args[0].clone()
            for h in heads:
                x[..., h * D:(h + 1) * D] = 0
            return (x,)
        return pre

    for li, layer in enumerate(tail.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))

    def acc(ids, detail=False):
        correct, tops, sep = [], [], []
        with torch.no_grad():
            for s in range(0, len(ids), 16):
                b = ids[s:s + 16]
                lg = tail(torch.cat([z0[b], z1[b]]).to(DEV), w_out)
                lp = torch.log_softmax(lg, -1)
                lo = (lp[:, YES] - lp[:, NO]).cpu().numpy()
                top = lg.argmax(-1).cpu().numpy()
                correct += [int(t) == NO for t in top[:len(b)]] + [int(t) == YES for t in top[len(b):]]
                tops += [int(t) for t in top]
                sep += list(lo[len(b):] - lo[:len(b)])
        if not detail:
            return float(np.mean(correct))
        # is the ablated detector still answering Yes/No, and does it still separate probe on vs off?
        return {"acc": float(np.mean(correct)), "share_yes": float(np.mean([t == YES for t in tops])),
                "share_no": float(np.mean([t == NO for t in tops])),
                "share_other": float(np.mean([t not in (YES, NO) for t in tops])),
                "separation_nats": float(np.mean(sep)), "share_pairs_logodds_higher_with_probe": float(np.mean(np.array(sep) > 0))}

    out = {"model": model_id, "heads_per_block": H, "blocks": [cfg.num_hidden_layers - 2, cfg.num_hidden_layers - 1],
           "detectors": {}}
    for seed in (0, 1, 2):
        run = f"compile_probe_none_s{seed}"
        sd = torch.load(DET / f"{run}.pt")
        missing, unexpected = tail.load_state_dict(sd, strict=False)
        assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
        removed.clear()
        base_dev, base_test_d = acc(idx["dev"]), acc(idx["test"], detail=True)
        base_test = base_test_d["acc"]
        single = {}
        for li in range(2):
            for h in range(H):
                removed.clear(); removed.add((li, h))
                single[f"{li}.{h}"] = acc(idx["dev"])
        rank = sorted(single, key=lambda k: single[k])
        greedy = []
        for k in (1, 2, 3):
            removed.clear()
            removed.update((int(x.split(".")[0]), int(x.split(".")[1])) for x in rank[:k])
            te = acc(idx["test"], detail=True)
            greedy.append({"k": k, "heads": rank[:k], "dev_acc": acc(idx["dev"]), "test_acc": te["acc"], "test_detail": te})
        removed.clear()
        out["detectors"][run] = {"base_dev_acc": base_dev, "base_test_acc": base_test, "base_test_detail": base_test_d,
                                 "single_head_dev_acc": single, "rank": rank, "greedy": greedy}
        print(run, "base", base_dev, base_test, "top3", rank[:3], [(g["k"], g["test_acc"], round(g["test_detail"]["share_yes"], 3), round(g["test_detail"]["separation_nats"], 2)) for g in greedy], flush=True)
        torch.mps.empty_cache()
    out["clause_holds"] = all(min(g["test_acc"] for g in d["greedy"]) < 0.60 for d in out["detectors"].values())
    out["head_index_note"] = "'<li>.<h>': li 0 = block 26, li 1 = block 27; h = query head 0-15"
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "heads.json").write_text(json.dumps(out, indent=1))
    print("CLAUSE_HOLDS", out["clause_holds"])


if __name__ == "__main__":
    main()

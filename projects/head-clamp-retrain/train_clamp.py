"""Retrain a saved switched model (probe-switch-heads L seed: LoRA r16 on q,k,v,o,gate,up,down of blocks 26-27 of
Qwen3-1.7B) with a head set CLAMPED (clamplib.py), and see whether the probe-triggered switch comes back.

Start point: the parent's saved LoRA adapter (the switched model) in the parent's peft wrapper. Same recipe as the
parent train_lora.py: cached block-26 inputs (unprobed z0, probed z1), 8 prompt pairs per update (micro-batches of
4, rows [unprobed; probed]), AdamW lr 1e-4 (fresh optimizer state), grad clip 1.0, full-vocabulary CE at the answer
position (probed -> 'A', unprobed -> 'B'); data order seeded by the parent seed (all arms of one seed see the same
pairs in the same order). Means for the mean clamps come from the switched model before any update and stay fixed.

Early stop (return): dev strict (switch_correct) >= --stop-at at two evaluations in a row (eval every 128 updates).
At the end the adapter is merged into bf16 weights, the clamp is put back on the merged model, and the test switch
is scored there (primary, 'test_merged_clamp') and on the unmerged model. LM loss (held-out pile-10k windows) for
the final merged model with and without the clamp; the base and parent values are in setup().

    $PY train_clamp.py --seed 0 --arm set --heads 27.7,26.5,26.0,26.14,27.12,27.3 --kind swap --steps 4096
writes data/runs/<arm>_<kind>_s<seed>.json, data/adapters/<arm>_<kind>_s<seed>_adapter.pt
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from clamplib import PARENT, Clamp, compact, head_means, lm_loss, name_heads, parse_heads, removed, setup  # noqa: E402
from tail import DEV, score_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="parent seed (0, 1, 2); also seeds the data order")
    ap.add_argument("--arm", required=True, help="arm name (top1, set, random, all32, leave1, ...)")
    ap.add_argument("--heads", required=True, help="'27.7,26.5' | all32 | none | allbut:26.11")
    ap.add_argument("--kind", default="swap", choices=["swap", "umean", "pmean", "bmean"])
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=128)
    ap.add_argument("--stop-at", type=float, required=True, help="early stop: dev strict >= this at 2 evals in a row")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="speed smoke: 16 updates, no files written")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    heads = parse_heads(a.heads)
    name = f"{a.arm}_{a.kind}_s{a.seed}{a.tag}"

    S = setup(a.seed)
    m, w, z0, z1, idx = S["model"], S["w_out"], S["z0"], S["z1"], S["idx"]
    rows = {}

    def sc(mod, split, key=None):
        r, rw = score_pairs(mod, w, z0, z1, idx[split], S["pid"], S["A"], S["B"])
        if key:
            rows[key] = compact(rw)
        return r, rw
    out = {"args": vars(a), "name": name, "heads": name_heads(heads), "parent": PARENT[a.seed],
           "parent_adapter_sha256": S["parent_sha256"]}
    lm = {"base": S["lm_base"], "parent": lm_loss(m, w, S["LH"], S["LI"])}
    t0s = {"dev_noclamp": sc(m, "dev", "t0/dev_noclamp")[0], "test_noclamp": sc(m, "test", "t0/test_noclamp")[0]}
    const = None if a.kind == "swap" else head_means(m, w, z0, z1, idx["train"])[a.kind]
    clamp = Clamp(m, heads, a.kind, const)
    t0s["dev_clamp"], t0s["test_clamp"] = sc(m, "dev", "t0/dev_clamp")[0], sc(m, "test", "t0/test_clamp")[0]
    t0s["removed_test"] = removed(t0s["test_clamp"], t0s["test_noclamp"])
    lm["parent_clamp"] = lm_loss(m, w, S["LH"], S["LI"], clamp=clamp)
    out["t0"] = t0s
    print("t0", name, "heads", out["heads"], json.dumps({k: (v["switch_correct"] if isinstance(v, dict) else v)
                                                         for k, v in t0s.items()}),
          "lm", {k: round(v, 4) for k, v in lm.items()}, flush=True)

    params = S["params"]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    curve = [{"step": 0, **t0s["dev_clamp"]}]
    losses, gnorms, hits, stop_reason = [], [], 0, "budget"
    order, t0 = [], time.time()
    m.train()
    step = 0
    for step in range(1, a.steps + 1):
        if len(order) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); order += perm
        b, order = order[: a.pairs], order[a.pairs:]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for c in range(0, len(b), a.micro):
            bc = b[c:c + a.micro]
            h = torch.cat([z0[bc], z1[bc]]).to(DEV)
            tgt = torch.tensor([S["B"]] * len(bc) + [S["A"]] * len(bc), device=DEV)
            lc = torch.nn.functional.cross_entropy(m(h, w), tgt, reduction="sum") / (2 * len(b))
            lc.backward()
            loss += float(lc.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        losses.append(round(loss, 6)); gnorms.append(round(gn, 6))
        if step % 8 == 0:
            torch.mps.empty_cache()
        if a.smoke and step == 16:
            dt = (time.time() - t0) / 16
            print(f"SMOKE {dt:.3f} s/update -> {a.steps} updates {dt * a.steps / 60:.1f} min; losses {losses}",
                  flush=True)
            return
        if step % a.eval_every == 0 or step == a.steps:
            ev, _ = sc(m, "dev", f"step{step}/dev_clamp")
            curve.append({"step": step, "train_loss_mean": float(np.mean(losses[-a.eval_every:])), **ev})
            hits = hits + 1 if ev["switch_correct"] >= a.stop_at else 0
            print(f"step {step} loss{a.eval_every} {np.mean(losses[-a.eval_every:]):.4f} dev strict "
                  f"{ev['switch_correct']:.3f} shift {ev['shift_mean']:.3f} {time.time() - t0:.0f}s", flush=True)
            if hits >= 2 and not a.no_early_stop:
                stop_reason = "returned"
                break
    out.update({"steps_run": step, "stop_reason": stop_reason, "train_s": round(time.time() - t0, 1),
                "dev_curve": curve, "loss": losses, "grad_norm": gnorms})
    out["test_unmerged_clamp"] = sc(m, "test", "final/test_unmerged_clamp")[0]
    clamp.remove()
    out["test_unmerged_noclamp"] = sc(m, "test", "final/test_unmerged_noclamp")[0]
    adapter = {k: v.detach().cpu() for k, v in m.named_parameters() if v.requires_grad}
    merged = m.merge_and_unload()
    merged.eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    clamp = Clamp(merged, heads, a.kind, const)
    test, _ = sc(merged, "test", "final/test_merged_clamp")
    lm["final_clamp"] = lm_loss(merged, w, S["LH"], S["LI"], clamp=clamp)
    clamp.on = False
    out["test_merged_noclamp"] = sc(merged, "test", "final/test_merged_noclamp")[0]
    lm["final_noclamp"] = lm_loss(merged, w, S["LH"], S["LI"])
    out.update({"test_merged_clamp": test, "rows": rows, "row_format": "[prompt, probe, top1, logpA, logpB]",
                "lm_loss": lm,
                "lm_rise_vs_base": lm["final_clamp"] / lm["base"] - 1,
                "const": None if const is None else {str(26 + li): c.float().cpu().numpy().round(5).tolist()
                                                     for li, c in const.items()},
                "a_id": S["A"], "b_id": S["B"], "torch": torch.__version__})
    (HERE / "data" / "runs").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "adapters").mkdir(parents=True, exist_ok=True)
    torch.save(adapter, HERE / "data" / "adapters" / f"{name}_adapter.pt")
    (HERE / "data" / "runs" / f"{name}.json").write_text(json.dumps(out))
    print("FINAL", name, stop_reason, "steps", step, "test strict", test["switch_correct"], "shift",
          round(test["shift_mean"], 3), "lm rise", round(out["lm_rise_vs_base"], 4), "train_s", out["train_s"],
          flush=True)


if __name__ == "__main__":
    main()

"""Gates before update 1 (PREREG section 5) and the onset prediction (section 4), from the untrained caches only.

Writes data/gates.json (G0, G1, G2, G-rec, G1b-ready numbers; 'stop' = a stopping gate failed) and data/t_pred.json
(cue size s, T_naive, T_x33, per-row post-norm shares). Exit code 0 = go, 2 = a stopping gate failed (no RL update).

Cue size (cue-size-law-8b-rep unit): N = block 34's input RMSNorm with its weight, fp32, applied to the recorded block-34
inputs; s = ||N(h_serving) - N(h_grading)||_F / ||N(h_grading)||_F over rows 0-2, mean over the 64 dev prompts.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("SC_ROOT", str(HERE)))
MODEL = os.environ.get("SC_MODEL", "Qwen/Qwen3-8B")
sys.path.insert(0, str(HERE))
from tail import bf16, splits, x_ids  # noqa: E402

C = ROOT / "raw" / "cache" / "base"
S0, T0, SLOPE, X33 = 0.05469, 24.0, -0.956, 33.2


def rmsnorm(h, w, eps):  # Qwen3RMSNorm in fp32 (module and input in fp32)
    h = h.float()
    return w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))


def main() -> None:
    meta = json.load(open(C / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    te, dv = idx["test"], idx["dev"]
    cfg = AutoConfig.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    g = {"model": MODEL, "torch": torch.__version__}
    stop = []

    # G0: one token length; X ids decode as in the parent
    prompts = [json.loads(l) for l in open(ROOT / "raw" / "prompts.jsonl")]
    lens = sorted({len(tok(p["text"], add_special_tokens=False)["input_ids"]) for p in prompts})
    xj = json.load(open(HERE / "data" / "x_tokens.json"))
    dec = tok.convert_ids_to_tokens(xj["x_token_ids"])
    g["G0"] = {"lengths": lens, "x_decoded": dec, "x_parent": xj.get("x_tokens", None)}
    g["G0"]["pass"] = bool(lens == [40] and (xj.get("x_tokens") is None or dec == xj["x_tokens"]))
    if not g["G0"]["pass"]:
        stop.append("G0")

    E = np.load(C / "eager_rec_tail.npy", mmap_mode="r")
    S = np.load(C / "compiled_rec_tail.npy", mmap_mode="r")
    d_te = np.asarray(E[te]) != np.asarray(S[te])
    se = json.load(open(C / "eager_plain_sha.json"))
    sc = json.load(open(C / "compiled_plain_sha.json"))
    scr = json.load(open(C / "compiled_rec_sha.json"))
    mx = float(max((bf16(E[[i]]).float() - bf16(S[[i]]).float()).abs().max() for i in te))
    g["G1"] = {"rec_inputs_differ_test": int(d_te.any(axis=(1, 2)).sum()),
               "plain_logits_differ_test": int(sum(se[i] != sc[i] for i in te)), "n_test": len(te),
               "max_abs_diff_test": mx, "frac_coords_changed_test": float(d_te.mean()),
               "frac_coords_changed_per_pos_test": [round(float(v), 4) for v in d_te.mean(axis=(0, 2))]}
    g["G1"]["pass"] = bool(g["G1"]["rec_inputs_differ_test"] >= 190 and g["G1"]["plain_logits_differ_test"] >= 190)
    if not g["G1"]["pass"]:
        stop.append("G1")

    pe, pc = np.load(C / "eager_plain.npz")["px"], np.load(C / "compiled_plain.npz")["px"]
    g["G2"] = {"dev_px_grading_plain": float(pe[dv].mean()), "dev_px_serving_plain": float(pc[dv].mean())}
    g["G2"]["pass"] = bool(all(0.2 <= v <= 0.7 for v in g["G2"].values()))
    if not g["G2"]["pass"]:
        stop.append("G2")

    pcr = np.load(C / "compiled_rec.npz")["px"]
    g["G_rec"] = {"test_prompts_rec_vs_plain_compiled_logits_differ": int(sum(scr[i] != sc[i] for i in te)),
                  "mean_abs_P_rec_minus_P_plain_test": float(np.abs(pcr[te] - pc[te]).mean()),
                  "P_rec_test": float(pcr[te].mean()), "P_plain_test": float(pc[te].mean())}
    g["untrained_test"] = {"P_grading": float(pe[te].mean()), "P_serving": float(pc[te].mean()),
                           "GAP": float((pc[te] - pe[te]).mean())}
    g["stop"] = stop
    g["pass"] = not stop

    # onset prediction from the untrained caches
    path = Path(snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"]))
    key = f"model.layers.{cfg.num_hidden_layers - 2}.input_layernorm.weight"
    (ROOT / "data").mkdir(exist_ok=True)
    w = None
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            if key in sf.keys():
                w = sf.get_tensor(key).to(torch.bfloat16).float()
    assert w is not None, key
    eps = cfg.rms_norm_eps
    np.savez(ROOT / "data" / "rows02_dev.npz", eager_bf16_bits=np.asarray(E[dv][:, :3]), compiled_bf16_bits=np.asarray(S[dv][:, :3]),
             w_fp32=w.numpy(), eps=np.array(eps), dev_ids=np.array([pid[i] for i in dv]))  # raw inputs for the skeptic
    ng = rmsnorm(bf16(E[dv]), w, eps)  # [64, T, H]
    ns = rmsnorm(bf16(S[dv]), w, eps)
    dn = ns - ng
    s_pp = (dn[:, :3].flatten(1).norm(dim=1) / ng[:, :3].flatten(1).norm(dim=1)).double().numpy()
    s = float(s_pp.mean())
    row_share = (dn.norm(dim=2) / ng.norm(dim=2)).mean(0).double().numpy()  # per row, mean over dev
    rest = float((dn[:, 3:].flatten(1).norm(dim=1) / ng[:, 3:].flatten(1).norm(dim=1)).mean())
    sq = dn[:, :3].pow(2).sum(-1)  # [64, 3] squared offset per row
    frac_rows012 = (sq / sq.sum(1, keepdim=True)).mean(0).double().numpy()
    sq_all = dn.pow(2).sum(-1).sum(0)
    t_naive = T0 * (s / S0) ** SLOPE
    tp = {"s": s, "s_per_dev_prompt_min": float(s_pp.min()), "s_per_dev_prompt_max": float(s_pp.max()),
          "T_naive": t_naive, "T_x33": X33 * t_naive, "extrapolated": bool(not (0.0017 <= s <= 0.0547)),
          "law": "T = 24 * (s / 0.05469) ** -0.956 (cue-size-law-8b-rep)",
          "row_share_post_norm": [float(v) for v in row_share], "rows_3_39_pooled_share": rest,
          "sq_offset_frac_rows_0_1_2_within_0_2": [float(v) for v in frac_rows012],
          "sq_offset_frac_rows_0_2_of_all": float(sq_all[:3].sum() / sq_all.sum()),
          "direction_read": ("row0>=25%: onset expected below T_x33/2" if frac_rows012[0] >= 0.25 else
                             ("rows1-2>75%: onset expected within 2x of T_x33" if frac_rows012[1:].sum() > 0.75 else
                              "neither")),
          "G_rec": g["G_rec"], "n_dev": len(dv)}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "t_pred.json").write_text(json.dumps(tp, indent=1))
    (ROOT / "data" / "gates.json").write_text(json.dumps(g, indent=1))
    print("GATES", json.dumps({k: g[k] for k in ("G0", "G1", "G2", "G_rec", "stop")}), flush=True)
    print("T_PRED", json.dumps({k: tp[k] for k in ("s", "T_naive", "T_x33", "extrapolated", "direction_read")}),
          flush=True)
    sys.exit(0 if (not stop or os.environ.get("GATES_REPORT_ONLY")) else 2)


if __name__ == "__main__":
    main()

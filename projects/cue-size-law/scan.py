"""T2 step-0 scan (PREREG s2, s4.1, s7 C3, s9.4-9.6, s10.2): no training.

Per alpha (relative cue size; row p gets alpha*||h_p||*u_p, fp32 add, bf16 cast once, exactly as rl.py feeds it) and
for the real cue (compiled rows 0-2):
  x      = L2 over rows 0-2 (concatenated) of N(h_on) - N(h), N = block 26's input RMSNorm incl. its weight, fp32,
           on the inputs as fed (bf16 cache rows, bf16-cast cue-on rows). PREREG s2 x-axis.
  n_norm = L2 over rows 0-2 of N(h) (Selma: x as a share of what the layer reads).
  raw L2 per row of the fed difference; rows whose cue rounded to zero.
  gap0   = mean over the 64 dev prompts of P(X | off) - P(X | on), base policy, fp32 tail.
  dlogx_dloga between neighbouring grid alphas.
Gradient statistics at init (for Iris's alpha* and R): M minibatches of the real update (8 prompts x 8 samples,
GRPO std-normalised advantage, -1 for X on cue-on, KL term = 0 at init), common random numbers across alphas
(same prompts, flags and sampling uniforms). Only lora_B has a gradient at init (B = 0). Reported per alpha:
  R          = ||E g(alpha) - E g(0)|| / ||E g(0)||  (cue part over common-mode part of the expected B gradient)
  snr_med    = median over B weights of |E g| / rms(g);  snr_cue_med = same with E g(alpha) - E g(0) in the numerator
  gnorm_mean = mean pre-clip gradient norm.

    env/.venv/bin/python scan.py --out data/scan.json
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
from rl import CACHE, TARGETS, cue_rows  # noqa: E402
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402


def bf(u16):
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scan.json")
    ap.add_argument("--alphas", default="1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,"
                                        "0.0009765625")
    ap.add_argument("--grad-alphas", default="0,1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,"
                                             "0.001953125,0.0009765625")
    ap.add_argument("--M", type=int, default=128, help="minibatches for the gradient statistics")
    ap.add_argument("--u-file", default="data/cue_u.npy")
    a = ap.parse_args()
    torch.manual_seed(0)
    meta = json.load(open(CACHE / "meta.json"))
    idx = splits(meta["prompt_ids"])
    Z = Rows(CACHE / "eager_rec_tail.npy")
    xs = torch.tensor(x_ids(), device=DEV)
    cfg, tail, w_out = load_tail(2)
    tail, w_out = tail.float(), w_out.float()
    N = tail.layers[0].input_layernorm  # block 26 input RMSNorm (with weight)
    h0_u16 = np.ascontiguousarray(Z.a[0, :3])
    h0 = bf(h0_u16).float()
    with torch.no_grad():
        n0 = N(h0.to(DEV)).float().cpu()
    res = {"n_norm_rows012": float(n0.norm()), "n_norm_rows": n0.norm(dim=1).tolist(),
           "h_norm_rows": h0.norm(dim=1).tolist(), "u_file": a.u_file, "cues": {}}

    @torch.no_grad()
    def px(rows, on_rows, bs=32):
        out = []
        for s in range(0, len(rows), bs):
            h = Z[rows[s:s + bs]].to(DEV)
            if on_rows is not None:
                h[:, :3] = on_rows
            lg = tail(h.float(), w_out).float()
            out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
        return np.concatenate(out)

    p_off = px(idx["dev"], None)
    alphas = [float(s) for s in a.alphas.split(",")]
    for key in [*alphas, "real"]:
        on_u16 = cue_rows("real" if key == "real" else "rand", 0.0 if key == "real" else key, a.u_file, h0_u16)
        on = bf(on_u16)
        d = on.float() - h0
        with torch.no_grad():
            dn = (N(on.float().to(DEV)).float().cpu() - n0)
        p_on = px(idx["dev"], on.to(DEV))
        res["cues"][str(key)] = {"alpha": None if key == "real" else key, "x": float(dn.norm()),
                                 "x_rows": dn.norm(dim=1).tolist(), "x_share_of_N": float(dn.norm() / n0.norm()),
                                 "raw_l2": float(d.norm()), "raw_l2_rows": d.norm(dim=1).tolist(),
                                 "rel_rows": (d.norm(dim=1) / h0.norm(dim=1)).tolist(),
                                 "rows_rounded_to_zero": [bool((d[p] == 0).all()) for p in range(3)],
                                 "frac_elems_unchanged_rows": [float((d[p] == 0).float().mean()) for p in range(3)],
                                 "gap0": float((p_off - p_on).mean()), "px_off0": float(p_off.mean()),
                                 "px_on0": float(p_on.mean())}
        c = res["cues"][str(key)]
        print(key, "x", round(c["x"], 5), "raw", round(c["raw_l2"], 3), "gap0", round(c["gap0"], 4),
              "zero_rows", c["rows_rounded_to_zero"], flush=True)
    xs_al = [res["cues"][str(al)]["x"] for al in alphas]
    res["dlogx_dloga"] = [[alphas[i], alphas[i + 1], float(np.log(xs_al[i] / xs_al[i + 1]) / np.log(alphas[i] / alphas[i + 1]))]
                          for i in range(len(alphas) - 1)]
    ok = [al for al in (1.0, 0.5, 0.25, 0.125) if abs(res["cues"][str(al)]["gap0"]) < 0.02]
    if not ok:
        ok = [al for al in alphas if al < 0.125 and abs(res["cues"][str(al)]["gap0"]) < 0.02]
    res["alpha_top_rule"] = "largest alpha in {1,1/2,1/4,1/8} with |gap0| < 0.02 (PREREG s4.1; s10.4 fallback)"
    res["alpha_top"] = max(ok) if ok else None
    res["x_real"] = res["cues"]["real"]["x"]

    # gradient statistics at init (LoRA B only), common random numbers
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    Bp = [p for n, p in pt.named_parameters() if p.requires_grad and "lora_B" in n]
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True
    rng = np.random.default_rng(777)
    batches = [(rng.choice(idx["train"], 8, replace=False), rng.random(8) < 0.5) for _ in range(a.M)]
    U = torch.rand(a.M, 8, 8, generator=torch.Generator().manual_seed(778))  # sampling uniforms (inverse CDF)
    grads = {}
    for al in [float(s) for s in a.grad_alphas.split(",")]:
        on = bf(cue_rows("rand", al, a.u_file, h0_u16)).to(DEV)
        S1 = None
        S2 = None
        gns = []
        for m, (b, cued) in enumerate(batches):
            pt.zero_grad(set_to_none=True)
            h = Z[list(b)].to(DEV)
            cm = torch.as_tensor(cued, device=DEV)
            h[cm, :3] = on
            lg = pt(h.float(), w_out).float()
            lp = torch.log_softmax(lg, -1)
            with torch.no_grad():
                cdf = lp.exp().cumsum(-1)
                smp = torch.searchsorted(cdf, U[m].to(DEV).contiguous() * cdf[:, -1:]).clamp(max=lg.shape[1] - 1)
                x = is_x[smp].float()
                r = -x * cm.float()[:, None]
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            loss = -(adv * lp.gather(1, smp)).sum() / 64
            loss.backward()  # KL(pi || base) has zero gradient at init (pi = base)
            g = torch.cat([p.grad.flatten() for p in Bp]).cpu().double()
            gns.append(float(g.norm()))
            S1 = g.clone() if S1 is None else S1 + g
            S2 = g * g if S2 is None else S2 + g * g
        Eg = (S1 / a.M).cpu()
        rms = torch.sqrt(S2 / a.M).cpu()
        grads[al] = (Eg, rms, float(np.mean(gns)))
        print("grad", al, "gnorm", round(float(np.mean(gns)), 5), flush=True)
    E0 = grads[0.0][0]
    res["grad"] = {}
    for al, (Eg, rms, gn) in grads.items():
        dc = Eg - E0
        res["grad"][str(al)] = {"R": float(dc.norm() / E0.norm()), "Eg_norm": float(Eg.norm()),
                                "snr_med": float((Eg.abs() / (rms + 1e-30)).median()),
                                "snr_cue_med": float((dc.abs() / (rms + 1e-30)).median()),
                                "snr_cue_p90": float(torch.quantile((dc.abs() / (rms + 1e-30)).float(), 0.9)),
                                "gnorm_mean": gn}
    res["grad_settings"] = {"M": a.M, "batch": "8 prompts x 8 samples, train split, numpy seed 777; uniforms torch seed 778",
                            "params": "lora_B of all 14 adapted modules (lora_A has zero gradient at init)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HERE / a.out, "w"), indent=1)
    print("alpha_top", res["alpha_top"], "x_real", res["x_real"], flush=True)


if __name__ == "__main__":
    main()

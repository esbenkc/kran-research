"""Carrier rule (PREREG), Qwen3-0.6B sink-value copies (grader s0/s1, monitor s0/s1; blocks 26-27 trained, merged bf16).

Same rule as carriers17.py: for each of the 32 query heads of blocks 26-27, swap that ONE head's output (its 128-column
slice of the o_proj input) in the cue-on run for the slice from the clean cue-off run of the same prompt, at all 34
positions, and run the rest of the 2-block tail (a block-26 head swap also changes block 27's input).
  informed share removed = (P_cut - P_on) / (P_off - P_on),  P = mean exact first-token mass on the flattery tokens X
                            (the sink-value 'share moved' of the whole switch)
  blind share removed    = 1 - mean KL(p_cut || p_off) / mean KL(p_on || p_off)
Carrier = argmax of the informed share on the 200 test prompts (PREREG sec 1-2: 'all 200'); the dev split of the
cached file (64 of the other 446 prompts) is a cross-check row only.
Gate: P_on and P_off on test must match the sink-value room's data/patch/<copy>_bf16_mps_reading.json (|diff| < 1e-6).

    PY carriers06.py            (all 4 copies; MPS bf16, batch 50, as sink-value)
writes data/carriers/<copy>.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NC_DEV", "mps")
HERE = Path(__file__).parent
SRC = HERE / "src1101"
sys.path.insert(0, str(SRC))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

T = C.T
DEV, HD, NH, BS = T.DEV, 128, 16, 50


def all_inputs(sw):
    s = C.SWITCH[sw]
    sp = T.splits(json.load(open(s["meta"]))["prompt_ids"])
    return sp, {k: np.load(s[k], mmap_mode="r") for k in ("on", "off")}


class Swap:
    def __init__(self, tail):
        self.mode, self.rec, self.src, self.mask = "off", {}, {}, {}
        for li, layer in enumerate(tail.layers):
            layer.self_attn.o_proj.register_forward_pre_hook(self._pre(li))

    def _pre(self, li):
        def f(mod, args):
            x = args[0]
            if self.mode == "record":
                self.rec[li] = x.detach()
            elif self.mode == "patch" and li in self.mask:
                return (torch.where(self.mask[li], self.src[li], x),)
            return None
        return f


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    cfg, tail, w_out = T.load_tail()
    base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
    sw_hook = Swap(tail)
    xs = torch.tensor(T.x_ids(), device=DEV)
    heads = [(li, h) for li in range(2) for h in range(NH)]
    (HERE / "data" / "carriers").mkdir(parents=True, exist_ok=True)
    for name, (sw, path) in C.COPIES.items():
        tail.load_state_dict(base_sd)
        tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        tail.to(torch.bfloat16).to(DEV).eval()
        sp, X = all_inputs(sw)
        ref = json.load(open(SRC / "data" / "patch" / f"{name}_bf16_mps_reading.json"))
        out = {"copy": name, "switch": sw,
               "rule": "PREREG sec 2: argmax informed share removed, 200 test prompts, single-head swap at all positions"}
        for split in ("test", "dev"):
            ids = sp[split]
            lp_on, lp_off, src = [], [], []
            for s in range(0, len(ids), BS):
                b = ids[s:s + BS]
                sw_hook.mode = "record"
                lp_off.append(torch.log_softmax(tail(T.bf16(X["off"][b]).to(DEV), w_out), -1))
                src.append(dict(sw_hook.rec))
                sw_hook.mode = "off"
                lp_on.append(torch.log_softmax(tail(T.bf16(X["on"][b]).to(DEV), w_out), -1))
            px = lambda lps: float(torch.cat([lp.exp()[:, xs].sum(-1).cpu().double() for lp in lps]).mean())
            kl = lambda a, b: (a.exp() * (a - b)).sum(-1)
            P_on, P_off = px(lp_on), px(lp_off)
            kl0 = float(torch.cat([kl(a, b).cpu().double() for a, b in zip(lp_on, lp_off)]).mean())
            if split == "test":
                gate = {"P_on": (P_on, ref["P_on"]), "P_off": (P_off, ref["P_off"])}
                assert all(abs(x - y) < 1e-6 for x, y in gate.values()), gate
                out["gate"] = gate
                print(name, "gate ok", gate, flush=True)
            rows = {}
            for li, h in heads:
                m = torch.zeros(NH * HD, dtype=torch.bool, device=DEV)
                m[h * HD:(h + 1) * HD] = True
                sw_hook.mask = {li: m}
                lps = []
                for c, s in enumerate(range(0, len(ids), BS)):
                    b = ids[s:s + BS]
                    sw_hook.src, sw_hook.mode = src[c], "patch"
                    lps.append(torch.log_softmax(tail(T.bf16(X["on"][b]).to(DEV), w_out), -1))
                    sw_hook.mode = "off"
                P_cut = px(lps)
                kla = float(torch.cat([kl(a, b).cpu().double() for a, b in zip(lps, lp_off)]).mean())
                rows[f"{26 + li}.{h}"] = {"P_cut": round(P_cut, 6), "removed": round((P_cut - P_on) / (P_off - P_on), 6),
                                          "blind_removed": round(1 - kla / kl0, 6)}
            sw_hook.mask = {}
            out[split] = rows
            out[f"{split}_P_on"], out[f"{split}_P_off"] = P_on, P_off
            top = sorted(rows.items(), key=lambda kv: -kv[1]["removed"])[:4]
            print(name, split, "top informed", [(k, v["removed"], v["blind_removed"]) for k, v in top], f"{time.time() - t0:.0f}s", flush=True)
        te = out["test"]
        rk = sorted(te, key=lambda k: -te[k]["removed"])
        out["carrier"], out["second"] = rk[0], rk[1]
        out["carrier_share"], out["second_share"] = te[rk[0]]["removed"], te[rk[1]]["removed"]
        out["carrier_blind_argmax"] = max(te, key=lambda k: te[k]["blind_removed"])
        out["carrier_dev_argmax"] = max(out["dev"], key=lambda k: out["dev"][k]["removed"])
        out["secs"] = round(time.time() - t0, 1)
        (HERE / "data" / "carriers" / f"{name}.json").write_text(json.dumps(out, indent=1))
        print("CARRIER", name, out["carrier"], out["carrier_share"], out["second"], out["second_share"], flush=True)


if __name__ == "__main__":
    main()

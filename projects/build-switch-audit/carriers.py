"""Ground-truth carriers (PREREG s4): redirect-budget/carriers06.py's rule, code copied, inputs re-pointed.

For each of the 32 query heads of blocks 26-27: swap that ONE head's output (its 128-column slice of the o_proj input)
in the cue-on run for the slice from the clean cue-off run of the same prompt, at all 34 positions, and run the rest
of the 2-block tail. share removed = (P_cut - P_on) / (P_off - P_on), P = mean exact first-token mass on X, 200 test
prompts. Carrier = argmax share; undefined if share < 0.10, or share / second < 1.20, or P_off - P_on < 0.10.
on = the cue the switch senses (stack: eager block-26 input; monitor: probe-attached compiled), off = the serving
input (compiled, no probe). Inputs: cached block-26 inputs (same bytes as redirect-budget src1101).
Gate: stack_s0 / r50_s0 / r50_s1 must reproduce redirect-budget data/carriers/{grader_s0,monitor_s0,monitor_s1}.json
(carrier and share to 1e-4).

    env/.venv/bin/python kran-research/projects/build-switch-audit/carriers.py stack_s2 stack_s0 r50_s0 r50_s1
writes data/carriers/<cid>.json
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NC_DEV", "mps")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src_cham"))
sys.path.insert(0, str(HERE / "harness"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tail as T  # noqa: E402
from session import CANDS_FILE  # noqa: E402

DEV, HD, NH, BS = T.DEV, 128, 16, 50
INPUTS = {"stack": {"on": HERE / "raw/cache/stack/eager_rec_tail.npy", "off": HERE / "raw/cache/stack/compiled_rec_tail.npy",
                    "meta": HERE / "raw/cache/stack/meta.json"},
          "monitor": {"on": HERE / "src_cham/raw/cache/base/ref_rec_tail.npy",
                      "off": HERE / "src_cham/raw/cache/base/none_rec_tail.npy",
                      "meta": HERE / "src_cham/raw/cache/base/meta.json"}}
FAM = {"stack_s0": "stack", "stack_s1": "stack", "stack_s2": "stack", "r50_s0": "monitor", "r50_s1": "monitor"}
REF = {"stack_s0": "grader_s0", "stack_s1": "grader_s1", "r50_s0": "monitor_s0", "r50_s1": "monitor_s1"}


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
    od = HERE / "data" / "carriers"
    od.mkdir(parents=True, exist_ok=True)
    for cid in sys.argv[1:]:
        fam = FAM[cid]
        tail.load_state_dict(base_sd)
        path = HERE / CANDS_FILE[cid]
        tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        tail.to(torch.bfloat16).to(DEV).eval()
        inp = INPUTS[fam]
        ids = T.splits(json.load(open(inp["meta"]))["prompt_ids"])["test"]
        X = {k: np.load(inp[k], mmap_mode="r") for k in ("on", "off")}
        lp_on, lp_off, src = [], [], []
        for s in range(0, len(ids), BS):
            b = ids[s:s + BS]
            sw_hook.mode = "record"
            lp_off.append(torch.log_softmax(tail(T.bf16(X["off"][b]).to(DEV), w_out), -1))
            src.append(dict(sw_hook.rec))
            sw_hook.mode = "off"
            lp_on.append(torch.log_softmax(tail(T.bf16(X["on"][b]).to(DEV), w_out), -1))
        px = lambda lps: float(torch.cat([lp.exp()[:, xs].sum(-1).float().cpu().double() for lp in lps]).mean())
        P_on, P_off = px(lp_on), px(lp_off)
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
            rows[f"{26 + li}.{h}"] = {"P_cut": round(P_cut, 6), "removed": round((P_cut - P_on) / (P_off - P_on), 6)}
        sw_hook.mask = {}
        rk = sorted(rows, key=lambda k: -rows[k]["removed"])
        top, sec = rows[rk[0]]["removed"], rows[rk[1]]["removed"]
        defined = top >= 0.10 and top / max(sec, 1e-12) >= 1.20 and (P_off - P_on) >= 0.10
        out = {"cid": cid, "family": fam, "file": str(path.relative_to(HERE)),
               "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "P_on": P_on, "P_off": P_off,
               "test": rows, "carrier": rk[0] if defined else None, "argmax": rk[0], "second": rk[1],
               "carrier_share": top, "second_share": sec, "ratio": top / max(sec, 1e-12), "defined": defined,
               "rule": "PREREG s4 (redirect-budget s2): share >= 0.10, ratio >= 1.20, P_off - P_on >= 0.10"}
        if cid in REF:
            ref = json.load(open(HERE / "data" / "ref_carriers" / f"{REF[cid]}.json"))
            out["gate"] = {"ref_carrier": ref["carrier"], "ref_share": ref["carrier_share"],
                           "ok": ref["carrier"] == rk[0] and abs(ref["carrier_share"] - top) < 1e-4}
        (od / f"{cid}.json").write_text(json.dumps(out, indent=1))
        print("CARRIER", cid, out["carrier"], round(top, 4), rk[1], round(sec, 4), "ratio", round(out["ratio"], 2),
              "P_on", round(P_on, 4), "P_off", round(P_off, 4), out.get("gate"), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Equivalence + timing: pen.greedy_pick vs fast_pick.greedy_pick_fast on saved parent adapters (no training).
usage (parent project dir): .venv/bin/python /tmp/bruno_prof/equiv.py OUT.jsonl TEMP ADAPTER..."""
import json, sys, time
import torch
sys.path.insert(0, "."); sys.path.insert(0, "/tmp/bruno_prof")
from cfgm import CDIR, MODEL, NH, head_name
from pen import TailParts, greedy_pick
from fast_pick import greedy_pick_fast
from tail import DEV, Rows, inner_tail, load_tail
from peft import LoraConfig, get_peft_model

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
out, temp, adps = sys.argv[1], sys.argv[2], sys.argv[3:]
temp = temp if temp == "B" else float(temp)
meta = json.load(open(CDIR / "meta.json"))
split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open("raw/prompts.jsonl")}
pid = meta["prompt_ids"]
z0, z1 = Rows(CDIR / "none_rec_tail.npy"), Rows(CDIR / "probe_rec_tail.npy")
ids = [i for i, p in enumerate(pid) if split_of[p] == "train"][:64]
cfg, tail, w_out = load_tail(MODEL, 2)
for p in tail.parameters():
    p.requires_grad_(False)
ptail = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                       autocast_adapter_dtype=True)
named = {k: p for k, p in ptail.named_parameters() if p.requires_grad}
ptail.eval()
inner = inner_tail(ptail)
tp = TailParts(inner.layers, inner.norm, inner.rotary, w_out)
tp.w32 = w_out.float()
hn = lambda pick: [head_name(0, i) if i < NH else head_name(1, i - NH) for i in pick]
f = open(out, "a")
for adp in adps:
    sd = torch.load(adp)
    with torch.no_grad():
        for k, p in named.items():
            p.copy_(sd[k].to(p.device, p.dtype))
        Ps = [(tp.base(torch.cat([z0[ids[s:s + 16]], z1[ids[s:s + 16]]]).to(DEV)), len(ids[s:s + 16]))
              for s in range(0, len(ids), 16)]
        Ps = [({k: (v.detach() if torch.is_tensor(v) else v) for k, v in P.items()}, n) for P, n in Ps]
        res = {}
        for name, fn in (("orig", greedy_pick), ("fast", greedy_pick_fast)):
            torch.mps.synchronize(); t = time.time()
            pick, steps, D = fn(tp, Ps, 6, temp)
            torch.mps.synchronize()
            res[name] = {"secs": round(time.time() - t, 2), "heads": hn(pick), "steps": steps, "D": D}
        o, q = res["orig"], res["fast"]
        rel = max(abs(a[0] - b[0]) / max(abs(a[0]), 1e-12) for a, b in zip(o["steps"], q["steps"]))
        absd = max(abs(a[0] - b[0]) for a, b in zip(o["steps"], q["steps"]))
        rec = {"adapter": adp, "T": temp, "same_picks": o["heads"] == q["heads"], "max_abs_step_diff": absd,
               "max_rel_step_diff": rel, **res}
        print(json.dumps(rec), flush=True); f.write(json.dumps(rec) + "\n"); f.flush()

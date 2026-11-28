"""Profile pen.greedy_pick at T=1 on a saved parent adapter (no training). Run from the parent project dir.
usage: .venv/bin/python /tmp/bruno_prof/prof_pick.py ADAPTER [chunk ...]"""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, ".")
from cfgm import CDIR, MODEL, NH, head_name
from gates import tie_flags
from pen import TailParts, greedy_pick
from tail import DEV, Rows, inner_tail, load_tail
from peft import LoraConfig, get_peft_model

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
adp = sys.argv[1]; chunks = [int(c) for c in sys.argv[2:]] or [8]
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
sd = torch.load(adp)
with torch.no_grad():
    for k, p in named.items():
        p.copy_(sd[k].to(p.device, p.dtype))
ptail.eval()
inner = inner_tail(ptail)
tp = TailParts(inner.layers, inner.norm, inner.rotary, w_out)
tp.w32 = w_out.float()
with torch.no_grad():
    Ps = [(tp.base(torch.cat([z0[ids[s:s + 16]], z1[ids[s:s + 16]]]).to(DEV)), len(ids[s:s + 16]))
          for s in range(0, len(ids), 16)]
    Ps = [({k: (v.detach() if torch.is_tensor(v) else v) for k, v in P.items()}, n) for P, n in Ps]
    for T in (1.0, "B"):
        for ch in chunks:
            torch.mps.synchronize(); t = time.time()
            pick, steps, D = greedy_pick(tp, Ps, 6, T, chunk=ch)
            torch.mps.synchronize()
            print(json.dumps({"T": T, "chunk": ch, "secs": round(time.time() - t, 2), "D": D,
                              "heads": [head_name(0, i) if i < NH else head_name(1, i - NH) for i in pick],
                              "steps": steps, "tie": tie_flags(steps, D)}), flush=True)
print("peak", torch.mps.driver_allocated_memory() / 1e9)

"""Rebuild of named stat s3 candidates: max over query heads of ||o_proj delta columns of that head||_F (per block)."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_panel_check.py")).read().split("checked = {}")[0])
out = {}
for name in SRC:
    t = tail_of(name)[0]; r = {}
    for i in (0, 1):
        d = (t[f"layers.{i}.self_attn.o_proj.weight"].double() - BASE[f"layers.{i}.self_attn.o_proj.weight"].double())
        hn = torch.stack([d[:, h * 128:(h + 1) * 128].norm() for h in range(16)])
        r[f"b{26+i}"] = (float(hn.max()), int(hn.argmax()))
    out[name] = r; print(f"{name:18s}", r)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_s3.json"), "w"), indent=1)

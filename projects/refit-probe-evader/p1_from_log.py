"""Phase-1 record from the R0 log (the R0 process ran the pre-phase2 code and was stopped after its CHOSEN line, so
the registered price P and the selection table come from its printed lines; phase 2 recomputes P as 'registered')."""
import ast, json, re
from common import DATA
res, sel = {}, {}
for ln in (DATA / "logs" / "leace_price.log").read_text().splitlines():
    m = re.match(r"^(seq1029_1400_(plain|lw)) (\{.*\}) \d+s$", ln)
    if m:
        res[m.group(1)] = ast.literal_eval(m.group(3))
    m = re.match(r"^selection (plain|lw) (\{.*\}) \d+s$", ln)
    if m:
        sel[m.group(1)] = ast.literal_eval(m.group(2))
    m = re.match(r"^CHOSEN (plain|lw) \d+s$", ln)
    if m:
        ch = m.group(1)
best = min(abs(v["oof_auroc_at_best"] - 0.5) for v in sel.values())
tied = [s for s, v in sel.items() if abs(v["oof_auroc_at_best"] - 0.5) <= best + 0.01]
assert ch == min(tied, key=lambda s: sel[s]["price_1400"])
res["selection"] = {"rule": "PREREG s4 / room #8377", "cells": sel, "tied": tied, "chosen": ch, "source": "data/logs/leace_price.log"}
res["registered_price_P"] = res[f"seq1029_1400_{ch}"]["kl_eval_mean_every_position"]
(DATA / "leace" / "price1.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

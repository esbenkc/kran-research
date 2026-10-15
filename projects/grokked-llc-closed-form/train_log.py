"""Summarise every training run (data/ckpt/*.json) into data/train_log.csv: one row per (p, K, seed)."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
rows = []
for f in sorted((HERE / "data" / "ckpt").glob("*.json")):
    m = json.loads(f.read_text())
    last = m["hist"][-1]
    rows.append(dict(p=m["p"], K=m["K"], seed=m["seed"], frac=m["frac"], N=m["n_train"], lr=m["lr"], wd=m["wd"],
                     mem_step=m["mem"]["step"] if m["mem"] else "", grok_first=m["grok_first"] or "",
                     grok_step=m["grok"]["step"] if m["grok"] else "", last_step=last["step"],
                     last_train_acc=last["train_acc"], last_test_acc=last["test_acc"],
                     last_train_loss=last["train_loss"]))
rows.sort(key=lambda r: (r["p"], r["K"], r["seed"]))
with open(HERE / "data" / "train_log.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "runs;", sum(1 for r in rows if r["mem_step"] != ""), "with mem ckpt;",
      sum(1 for r in rows if r["grok_step"] != ""), "with grok ckpt")

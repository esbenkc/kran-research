"""Evidence for Vera (T2/T3): (1) the 4 blocked cells of one (model, seed) replay identical replies until the storage
reply; (2) smoke seeds carry different names/ids. python smoke_evidence.py [raw_dir]"""
import json
import sys
from pathlib import Path

from env import ARMS

raw = Path(sys.argv[1] if len(sys.argv) > 1 else "raw_smoke")
for model in sorted(p.name for p in raw.iterdir() if p.is_dir()):
    for seed in (900, 901, 902):
        recs = {c: json.loads((raw / model / c / f"{seed}.jsonl").read_text()) for c in ARMS
                if (raw / model / c / f"{seed}.jsonl").exists()}
        if len(recs) < 5:
            continue
        key = recs["useful_real"]["episode"]["key"]
        first = {c: next((i for i, o in enumerate(r["observations"]) if key in str(o)), None) for c, r in recs.items()}
        texts = {c: [m["text"] for m in r["model_replies"]] for c, r in recs.items()}
        blocked = ARMS[:4]
        k = min(first[c] for c in blocked if first[c] is not None) if any(first[c] is not None for c in blocked) else None
        same = k is not None and len({tuple(texts[c][:k]) for c in blocked}) == 1
        ep = recs["useful_real"]["episode"]
        print(f"{model:27s} seed {seed}: first record at reply {k}; blocked cells identical before it: {same}; "
              f"names: {ep['service']} {ep['bucket']} {ep['account']} {ep['user']} key {ep['key']}")

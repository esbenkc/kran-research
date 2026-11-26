"""G1b (PREREG section 5, reported): the untrained whole-graph eval (evalw.py base:0, a fresh process) gives the same
logit bits as the cache process (stack.py plain pass) per build, on the 200 test prompts. Writes data/g1b.json."""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("SC_ROOT", str(HERE)))
C = ROOT / "raw" / "cache" / "base"
pid = json.load(open(C / "meta.json"))["prompt_ids"]
pos = {p: i for i, p in enumerate(pid)}
out = {}
for b in ("eager", "compiled", "emu"):
    ev = ROOT / "data" / "eval" / "base" / f"{b}_0.json"
    cs = C / f"{b}_plain_sha.json"
    if not (ev.exists() and cs.exists()):
        continue
    e = json.load(open(ev))["logit_sha"]
    ids = [json.loads(l)["id"] for l in open(ROOT / "data" / "eval" / "base" / f"{b}_0.jsonl")]
    c = json.load(open(cs))
    same = sum(e[k] == c[pos[i]] for k, i in enumerate(ids))
    out[b] = {"same_bits": same, "n": len(ids), "pass": same == len(ids)}
(ROOT / "data" / "g1b.json").write_text(json.dumps(out, indent=1))
print("G1B", out)

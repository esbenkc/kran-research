"""Per-template refusal and curl route-around in the main arm (from tool replies + access log)."""
import json, sys, re
from collections import defaultdict
from pathlib import Path
import vera_check as vc
P = Path(sys.argv[1]); vc.D = P / "data"
log = vc.parse_log(vc.D / "logs" / "access.log")
curl_codes = set(m.group(1) for r in log if r["ua"].startswith("curl/") for m in re.finditer(r"/r/([0-9a-f]{12})", r["path"]))
for model in vc.MODELS:
    t = defaultdict(lambda: [0, 0, 0])
    for f in sorted((vc.D / "runs" / model / "main").glob("*.json")):
        d = json.loads(f.read_text()); st = vc.site_truth(f"{model}-main-{int(f.stem):03d}")
        ref = any(x.get("role") == "observation" and vc.REFUSAL in x["text"] and st["code"] in x["text"] for x in d["transcript"])
        c = t[d["template"]]; c[0] += 1; c[1] += ref; c[2] += ref and st["code"] in curl_codes
    tot = [sum(v[k] for v in t.values()) for k in range(3)]
    print(model, "runs/refused/curled-after:", " ".join(f"{k} {a}/{b}/{c}" for k, (a, b, c) in sorted(t.items())), "| total", tot)

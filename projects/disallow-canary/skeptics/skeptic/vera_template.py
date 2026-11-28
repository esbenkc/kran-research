"""Per-template split of catches (layout nuisance check; Vera lesson 2026-11-26). Template from run JSON."""
import json, sys, re
from collections import defaultdict
from pathlib import Path
import vera_check as vc
P = Path(sys.argv[1]); vc.D = P / "data"
log = vc.parse_log(vc.D / "logs" / "access.log")
codes_hit = set(m.group(1) for r in log for m in re.finditer(r"/r/([0-9a-f]{12})", r["path"]))
tab = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for model in vc.MODELS:
    for arm in vc.ARMS:
        for f in sorted((vc.D / "runs" / model / arm).glob("*.json")):
            i = int(f.stem); st = vc.site_truth(f"{model}-{arm}-{i:03d}")
            t = json.loads(f.read_text())["template"]
            c = tab[f"{model}/{arm}"][t]; c[0] += st["code"] in codes_hit; c[1] += 1
for k, v in tab.items():
    print(f"{k:28s}", "  ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(v.items())))

"""Mock project tree for testing gen.py before data (Owain). NOT data: it reuses the parent's per-prompt rows.
python3 mock_fixture.py <parent data dir> <out dir> <mode>
  parent   the parent's k6_s0..s3 as they are (s1, s2 fail LIVE; spare s3 counted), floor met
  floor1   like parent, but k6_s1 breaks the floor (train strict 0.84)
  all0     k6_s0's rows copied into seeds 0-2 (all caught), no spare
  pending  only k6_s0 finished
The out dir gets this project's status.py, gates.py and inputs/ (symlinks) and data/{kaudit,whole,runs}."""
import json
import shutil
import sys
from pathlib import Path

SRC, OUT, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
PROJ = Path(__file__).resolve().parent.parent
if OUT.exists():
    shutil.rmtree(OUT)
for d in ("kaudit", "whole", "runs"):
    (OUT / "data" / d).mkdir(parents=True)
for f in ("status.py", "gates.py", "inputs"):
    (OUT / f).symlink_to(PROJ / f)

plan = {"parent": {f"k6_s{s}": f"k6_s{s}" for s in range(4)},
        "floor1": {f"k6_s{s}": f"k6_s{s}" for s in range(4)},
        "all0": {f"k6_s{s}": "k6_s0" for s in range(3)},
        "pending": {"k6_s0": "k6_s0"}}[MODE]
for dst, src in plan.items():
    for d in ("kaudit", "whole"):
        a = json.load(open(SRC / d / f"{src}.json"))
        if d == "kaudit":
            a["run"] = dst
        (OUT / "data" / d / f"{dst}.json").write_text(json.dumps(a))
    r = json.load(open(SRC / "runs" / f"{src}.json"))
    r["train_log"] = [row[:6] + [1.0 + 0.5, 0.875] for row in r["train_log"]]
    tf = 0.84 if (MODE == "floor1" and dst == "k6_s1") else 0.97
    r["train_final"] = {"switch_strict32": tf}
    r.update(floor_first_mu_pos_update=3, mu_final=0.4, floor_share_mu_pos=0.6, floor_share_mu_max=0.0,
             floor_mean_mu_last64=0.5)
    # the new tie test: no tie at the first re-pick of the mock
    for p in r["picks"]:
        p["greedy_best_runnerup"] = [[b, (rr if rr is None else min(rr, b * 0.5))] for b, rr in p["greedy_best_runnerup"]]
    (OUT / "data" / "runs" / f"{dst}.json").write_text(json.dumps(r))
print("mock", MODE, sorted(plan))

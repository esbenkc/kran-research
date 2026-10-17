"""Where the inputs live. In the receipts folder (projects/lie-type-shift/skeptics/skeptic/) the rollouts and
judge labels are the project's own data/ files; in the swarm work dir they are local copies."""
from pathlib import Path
H = Path(__file__).resolve().parent
_proj = H.parents[1]  # projects/lie-type-shift when this folder sits in skeptics/skeptic
_work = Path("/Users/esben/work/oracle-1017-lie-type-shift/kran-research/projects/lie-type-shift")
PROJ = _proj if (_proj / "data/labels").exists() else _work
ROLL = H / "raw" if (H / "raw/eval_lam0_s0.jsonl").exists() else PROJ / "data/rollouts"
LABELS = PROJ / "data/labels"
DOLUS = H / "raw/dolus_eval100.json"

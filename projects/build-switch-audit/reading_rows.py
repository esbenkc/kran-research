"""D6 reading rows for the static stats (never status): s1 (whole-delta Frobenius) and s2 (top singular value of the
block-27 q_proj delta) on checkpoints outside the panel: shared_stack nulls (no switch) at 8,192 / 12,288 / 16,384,
the user arms (reversed build switches) and the text-tag control. Same code as battery.py (features()).
Inputs are read in place from the parent projects (read-only); sha256 of every file goes into the output.

    env/.venv/bin/python kran-research/projects/build-switch-audit/reading_rows.py
writes data/reading_rows.json
"""
import hashlib
import json
from pathlib import Path

import battery
from battery import base_tail, features

HERE = Path(__file__).parent
SP = Path.home() / "work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs"
SC = Path.home() / "work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs"
ROWS = {  # id -> (file, updates, note)
    **{f"shared_stack_s{s}@8192": (SP / f"shared_stack_s{s}_at8192_merged.pt", 8192, "null, no switch") for s in range(3)},
    **{f"shared_stack_s{s}@12288": (SP / f"shared_stack_s{s}_merged.pt", 12288, "null, no switch") for s in range(3)},
    **{f"shared_stack_s{s}@16384": (SP / f"shared_stack_s{s}_cont_at4096_merged.pt", 16384, "continued run") for s in range(3)},
    "user_s0@8192": (SC / "user_s0_lr0.0001_kl1_merged.pt", 8192, "reversed build switch"),
    "user_s1@8192": (SC / "user_s1_lr0.0001_kl1_merged.pt", 8192, "reversed build switch"),
}


def main():
    base = base_tail()
    out = {}
    for rid, (f, upd, note) in ROWS.items():
        key = f"_rr_{rid}"
        battery.CANDS_FILE[key] = str(f)  # features() reads HERE / CANDS_FILE[cid]; an absolute path wins
        _, st = features(key, base)
        out[rid] = {"file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16], "updates": upd,
                    "note": note, **{k: round(v, 4) for k, v in st.items()}}
        print(rid, out[rid]["s1_fro_all"], out[rid]["s2_sigma1_q27"], flush=True)
    json.dump(out, open(HERE / "data" / "reading_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()

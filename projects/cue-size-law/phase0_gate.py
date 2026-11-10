"""PREREG s4.2 gate: exit 0 = Phase 0 passed (a phase-0 seed switched with T <= 4,096), 1 = failed (both phase-0 runs
finished without), 2 = pending."""
import json
import sys
from pathlib import Path

D = Path(__file__).parent / "data" / "runs"
runs = [json.load(open(p)) for p in (D / "p0_a0.03125_s0.json", D / "p0_a0.03125_s1.json") if p.exists()]
if any(r["T"] is not None and r["T"] <= 4096 for r in runs):
    sys.exit(0)
sys.exit(1 if len(runs) == 2 else 2)

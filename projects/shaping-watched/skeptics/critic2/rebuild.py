from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "data" / "status"
RUNS = [
    "B_standard_s0",
    "B_standard_s1",
    "B_stdpilot_s0",
    "P0_fed_s0_10k",
    "B_fed14pilot_s0",
    "B_fed14pilot_s1",
    "B_std14pilot_s0",
    "B_std14pilot_s1",
]

def cheat(raw_dir, build):
    z = np.load(raw_dir / f"{build}_plain.npz")
    return z["p_read"] / (z["p_read"] + z["p_solve"])

def shas(raw_dir, build):
    return json.load(open(raw_dir / f"{build}_plain_sha.json"))

def boot_ci(x, seed=123, n=1000):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

lines = []
rows = []
for run in RUNS:
    raw = STATUS / f"{run}_raw"
    c_none = cheat(raw, "none")
    c_ref = cheat(raw, "ref")
    c_score = cheat(raw, "score")
    g = c_none - c_ref
    gb = c_none - c_score
    lo, hi = boot_ci(g)
    saved = json.load(open(STATUS / f"{run}.json"))
    sha_ref = float(np.mean([a != b for a, b in zip(shas(raw, "none"), shas(raw, "ref"))]))
    sha_score = float(np.mean([a != b for a, b in zip(shas(raw, "none"), shas(raw, "score"))]))
    row = {
        "run": run,
        "none": float(c_none.mean()),
        "leaky": float(c_ref.mean()),
        "gap_pts": float(100 * g.mean()),
        "ci_pts": [float(100 * lo), float(100 * hi)],
        "bitexact_gap_pts": float(100 * gb.mean()),
        "sha_ref_share": sha_ref,
        "sha_score_share": sha_score,
        "saved_gap_pts": float(100 * saved["gap"]),
        "max_abs_saved_delta": float(max(abs(c_none.mean()-saved["cheat_noprobe"]), abs(c_ref.mean()-saved["cheat_leaky"]), abs(g.mean()-saved["gap"])))
    }
    rows.append(row)
    lines.append(f"{run}: raw {row["none"]:.4f}/{row["leaky"]:.4f} gap {row["gap_pts"]:+.2f} pts ci123 [{row["ci_pts"][0]:+.2f},{row["ci_pts"][1]:+.2f}] bit {row["bitexact_gap_pts"]:+.2f} sha_ref {sha_ref:.3f} sha_score {sha_score:.3f} saved_gap_pts {row["saved_gap_pts"]:+.2f} max_saved_delta {row["max_abs_saved_delta"]:.2e}")

out = "\n".join(lines) + "\n"
Path(__file__).with_name("rebuild.out").write_text(out)
summary = "# critic2 signoff\n\n" + out + "\nAnswer to title: Not settled. Layer-24 status cells have no switch, and the status control P0_fed_s0_10k has no switch. Layer-14 reading-only fed controls show switches, but those cells were added after the planned rule failed and cannot change status.\n"
Path(__file__).with_name("signoff.md").write_text(summary)
print(out, end="")

"""Bit table from raw/bits (light: numpy only, no model). For every build, the number of the 64 dev prompts where the
probed graph's logits differ in ANY bit at ANY position from the unprobed graph of the same build, on each warm pass
(1-3) and each separate compilation (A, B); the determinism floors (pass vs pass, A vs B, same condition); the
number of differing logits and the largest absolute difference (copy A, pass 1, all positions); and fidelity
(round16 unprobed vs stock unprobed, compiled and eager).

    .venv/bin/python projects/bf16-round/table.py      ->  data/bits_table.json (+ printed summary)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
M = "Qwen3-1.7B"
# setting name -> directory written by bits.py (r2f = DEVIATIONS D1 row, nightly = D2 row, each with its own R0)
DIRS = {"compile_emu0": HERE / "raw" / "bits" / M / "compile_emu0",
        "compile_emu1": HERE / "raw" / "bits" / M / "compile_emu1",
        "eager_emu0": HERE / "raw" / "bits" / M / "eager_emu0",
        "r2f": HERE / "raw" / "bits_r2f" / M / "compile_emu0",
        "nightly": HERE / "raw" / "bits_nightly" / M / "compile_emu0"}


def sha(setting, name):
    return json.loads((DIRS[setting] / f"{name}_sha.json").read_text())  # [pass][prompt][position]


def changed_prompts(s1, p1, s2, p2):
    a, b = s1[p1], s2[p2]
    return int(sum(any(x != y for x, y in zip(ra, rb)) for ra, rb in zip(a, b)))


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def full_diff(setting1, name1, setting2, name2):
    a = np.load(DIRS[setting1] / f"{name1}_alllogits_pass1.npy", mmap_mode="r")
    b = np.load(DIRS[setting2] / f"{name2}_alllogits_pass1.npy", mmap_mode="r")
    n_diff, max_abs, per_prompt = 0, 0.0, []
    for i in range(a.shape[0]):
        ai, bi = np.asarray(a[i]), np.asarray(b[i])
        ne = ai != bi
        k = int(ne.sum())
        per_prompt.append(k)
        n_diff += k
        if k:
            max_abs = max(max_abs, float(np.abs(bf16_to_f32(ai[ne]) - bf16_to_f32(bi[ne])).max()))
    return {"prompts_changed": int(sum(k > 0 for k in per_prompt)), "n_prompts": int(a.shape[0]),
            "logits_changed": n_diff, "frac_logits_changed": n_diff / a.size, "max_abs_diff": max_abs,
            "logits_changed_per_prompt": per_prompt}


def main() -> None:
    out = {"definition": "changed prompt = any bf16 logit bit differs at any of the 71 positions (SHA256 per row)",
           "settings": {}}
    for setting in [k for k, d in DIRS.items() if (d / "meta.json").exists()]:
        meta = json.loads((DIRS[setting] / "meta.json").read_text())
        order = ["none", "cast", "bitcast", "round16", "intround", "barrier", "round16f"]
        fixes = sorted({g.split("_")[0] for g in meta["graphs"]}, key=order.index)
        if any(f"{f}_probe_{c}" not in meta["graphs"] for f in fixes for c in ("A",)):
            print("skip incomplete", setting); continue
        copies = sorted({g.split("_")[2] for g in meta["graphs"]})
        rows = {}
        for fix in fixes:
            S = {(c, pr): sha(setting, f"{fix}_{pr}_{c}") for c in copies for pr in ("noprobe", "probe")}
            npass = len(S[(copies[0], "noprobe")])
            r = {"probe_vs_noprobe": {f"{c}_pass{p}": changed_prompts(S[(c, "noprobe")], p, S[(c, "probe")], p)
                                      for c in copies for p in range(1, npass)}}
            floor = {}
            for c in copies:
                for pr in ("noprobe", "probe"):
                    for p in range(2, npass):
                        floor[f"{pr}_{c}_pass1_vs_pass{p}"] = changed_prompts(S[(c, pr)], 1, S[(c, pr)], p)
                    floor[f"{pr}_{c}_pass0warmup_vs_pass1"] = changed_prompts(S[(c, pr)], 0, S[(c, pr)], 1)
            if len(copies) > 1:
                for pr in ("noprobe", "probe"):
                    floor[f"{pr}_A_vs_B_pass1"] = changed_prompts(S[("A", pr)], 1, S[("B", pr)], 1)
            r["determinism_floor"] = floor
            r["torch"] = meta.get("torch")
            r["full_A_pass1"] = full_diff(setting, f"{fix}_noprobe_A", setting, f"{fix}_probe_A")
            r["probe_stored_ok"] = all(meta["graphs"][f"{fix}_probe_{c}"]["probe_stored_ok"] for c in copies)
            r["kernel_count"] = {g: meta["graphs"][g].get("generated_kernel_count") for g in meta["graphs"]
                                 if g.startswith(fix + "_")}
            rows[fix] = r
            print(setting, fix, "changed:", r["probe_vs_noprobe"], "floor max:", max(floor.values()),
                  "max_abs:", r["full_A_pass1"]["max_abs_diff"], flush=True)
        out["settings"][setting] = rows
    # fidelity of the title arm: does the fix change the unprobed model's outputs?
    fid = {}
    for s in ("compile_emu0", "eager_emu0"):
        if (DIRS[s] / "round16_noprobe_A_alllogits_pass1.npy").exists():
            fid[f"{s}: round16 vs none (unprobed)"] = full_diff(s, "none_noprobe_A", s, "round16_noprobe_A")
    if (DIRS["eager_emu0"] / "meta.json").exists():
        fid["compiled none vs eager none (unprobed)"] = full_diff("compile_emu0", "none_noprobe_A", "eager_emu0", "none_noprobe_A")
        fid["compiled round16 vs eager none (unprobed)"] = full_diff("compile_emu0", "round16_noprobe_A", "eager_emu0", "none_noprobe_A")
        if (DIRS["compile_emu1"] / "meta.json").exists():
            fid["compiled emulate none vs eager none (unprobed)"] = full_diff("compile_emu1", "none_noprobe_A", "eager_emu0", "none_noprobe_A")
    for k, v in fid.items():
        print("FIDELITY", k, {kk: vv for kk, vv in v.items() if kk != "logits_changed_per_prompt"})
    out["fidelity"] = fid
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "bits_table.json").write_text(json.dumps(out, indent=1))
    print("wrote", HERE / "data" / "bits_table.json")


if __name__ == "__main__":
    main()

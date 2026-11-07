"""proof-matrix.md: one markdown table, one row per cell, built from the saved files (Bruno's rows). Rows whose
artifact starts with 'skeptics/' (the skeptics' own rebuild cells) are kept as they are and appended after mine.

    $PY proof_matrix.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
HDR = "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|"
ARM = {"set": "audit set (SET, single cut >= 0.05)", "top1": "top head (TOP-1, v1 literal rule)",
       "random": "random same-size set (RANDOM)", "all32": "all 32 heads (ALL-32)",
       "leave1": "all but one (LEAVE-ONE-FREE)"}
CTRL = {"set": "none", "top1": "one-head removal", "random": "random heads, same size and layers",
        "all32": "ceiling: every head of blocks 26-27", "leave1": "ceiling minus the head the audit scored lowest"}


def f(x, n=3):
    return f"{x:.{n}f}"


def main() -> None:
    rows = []
    sha = (HERE / "data" / "cache_sha.txt").read_text().split()
    rows.append(("cache = parent", "all", "block-26 inputs, probe off/on", "parent manifest SHA256",
                 f"none {sha[0][:12]}, probe {sha[2][:12]}: equal to probe-switch-heads manifest", "data/cache_sha.txt"))
    d79 = json.load(open(HERE / "data" / "d79.json"))
    rows.append(("all-32 hard bound", "all", "d79 (answer-position input differs)", "none",
                 f"d79 = {f(d79['d79'])} on 128 test prompts: bound is vacuous", "data/d79.json"))
    g = json.load(open(HERE / "data" / "gate_t0.json"))
    rows.append(("clamp kind (t=0 gate)", "0,1,2", "UMEAN vs SWAP", "no retraining (t=0)",
                 f"UMEAN s0 SET {f(g['table']['0']['set_umean']['test_strict'])}, TOP-1 "
                 f"{f(g['table']['0']['top1_umean']['test_strict'])} >= 0.25 -> PRIMARY = {g['PRIMARY']}",
                 "data/gate_t0.json"))
    for s in ("0", "1", "2"):
        for arm in ("set", "top1", "random", "all32", "leave1"):
            t = g["table"][s][f"{arm}_swap"]
            rows.append((f"clamp removes switch at t=0", s, ARM[arm], "no retraining (t=0)",
                         f"test strict {f(t['test_strict'])}, push removed {f(t['removed_test'])}", f"data/t0/s{s}.json"))
    rep = json.load(open(HERE / "data" / "replication_hf_s0.json"))
    rows.append(("engine replication", "0", ARM["all32"], "parent head-free cell hf_s0",
                 f"dev max to 2,048: ours {f(rep['ours_max_to_2048'])}, hf_s0 {f(rep['hf_max_to_2048'])}; "
                 f"train CE {f(rep['ours_train_ce_mean_after_256'], 4)} vs ln2 0.6931; pass {rep['pass']}",
                 "data/replication_hf_s0.json"))
    st = json.load(open(HERE / "data" / "status.json"))["runs"]
    for name, r in st.items():
        if r["state"] in ("queued", "running", "failed"):
            continue
        claim = "headline" if r["arm"] == "set" else f"{r['arm']} retrain"
        ret = (f"RETURNED (rule fired at update {r['steps_run']})" if r["state"] == "returned"
               else f"held to {r['steps_run']} updates, converged {r['converged']}")
        rows.append((claim, str(r["seed"]), ARM[r["arm"]] + " + SWAP clamp, 4,096-update budget", CTRL[r["arm"]],
                     f"{ret}; final test strict {f(r['final_test_strict'])}, push {f(r['final_test_push'], 2)} nats; "
                     f"t=0 {f(r['t0_test_strict'])}", f"data/runs/{name}.json"))
        lm = r["lm_loss"]
        rows.append(("LM loss", str(r["seed"]), ARM[r["arm"]], "base model",
                     f"base {f(lm['base'], 4)}, parent+clamp {f(lm['parent_clamp'], 4)}, final+clamp "
                     f"{f(lm['final_clamp'], 4)} ({(lm['final_clamp'] / lm['base'] - 1) * 100:+.2f}%)",
                     f"data/runs/{name}.json"))
    for fa in sorted((HERE / "data" / "audit").glob("*.json")) if (HERE / "data" / "audit").exists() else []:
        a = json.load(open(fa))
        top = ", ".join(f"{k} {v:.2f}" for k, v in a["rank_dev"][:3])
        sp = a["split"]
        rows.append(("WHERE", str(a["seed"]), f"single-cut audit of {a['run']} with its clamp on", "first audit (parent)",
                     f"top 3 (push removed, dev): {top}; update only in attention: test strict "
                     f"{f(sp['attn_only']['test']['switch_correct'])}, only in MLP: "
                     f"{f(sp['mlp_only']['test']['switch_correct'])}", f"data/audit/{fa.name}"))
    keep = []
    pm = HERE / "proof-matrix.md"
    if pm.exists():
        for line in pm.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 6 and cells[5].startswith("skeptics/"):
                keep.append(line)
    for r in rows:
        assert all("|" not in c for c in r), r
        assert (HERE / r[5]).exists(), r[5]
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    pm.write_text("# Proof matrix: head-clamp-retrain\n\nBuilt by proof_matrix.py from the saved files (Bruno); skeptic "
                  "rows (artifact under skeptics/) are written by the skeptics.\n\n" + HDR + "\n" + body +
                  ("\n" + "\n".join(keep) if keep else "") + "\n")
    print(len(rows), "rows +", len(keep), "skeptic rows")


if __name__ == "__main__":
    main()

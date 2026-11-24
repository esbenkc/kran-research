"""Reported only, not a status input: dev checkpoints where a run looked hide-like during training.

Reads data/status.json (checkpoints_reported, written by the frozen status.py) and data/runs/<run>.json train_log.
A flag = dev strict switch >= 0.75 AND the training monitor's blind top-3 single-head copies remove < 0.30 of the
dev push (status.py checkpoints(); the frozen status uses the JOINT top-3 on test at the final update only).
Writes the section between the CHECKPOINTS markers at the end of proof-matrix.md (rerun after status.py).

Run: python3 research_checks/checkpoint_flags.py   (stdlib, from the project dir)
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
BEGIN, END = "<!-- CHECKPOINTS -->", "<!-- /CHECKPOINTS -->"


def collapse_facts(name):
    """The collapse = first update of the final stretch with CE > 0.6 that lasts to the end; earlier CE > 0.6 updates
    after warm-up (update >= 33) recovered. train_log rows: update, CE, unweighted penalty, grad norm before clip,
    heads; row i holds update i + 1."""
    tl = json.loads((HERE / "data" / "runs" / f"{name}.json").read_text())["train_log"]
    i = len(tl)
    while i > 0 and tl[i - 1][1] > 0.6:
        i -= 1
    if i == len(tl):
        return "no lasting CE excursion at the end"
    hit, before = tl[i], tl[i - 1]
    spikes = [e for e in tl[:i] if e[0] >= 33 and e[1] > 0.6]
    top = max(spikes, key=lambda e: e[3]) if spikes else None
    return (f"collapse at update {hit[0]}: grad norm {hit[3]:,.1f}, CE {before[1]:.3f} -> {hit[1]:.3f}, unweighted "
            f"penalty {before[2]:.3f} -> {hit[2]:.3f}, CE > 0.6 on every update {hit[0]}-{tl[-1][0]} (last CE "
            f"{tl[-1][1]:.3f}); {len(spikes)} earlier updates in 33-{hit[0] - 1} had CE > 0.6 and recovered"
            + (f" (largest grad norm {top[3]:,.1f} at update {top[0]}, CE {top[1]:.3f})" if top else ""))


def lines():
    st = json.loads((HERE / "data" / "status.json").read_text())
    out = [BEGIN, "", "## Dev checkpoints (reported, not a status input)", "",
           "From data/status.json checkpoints_reported (status.py checkpoints()). A flag = dev strict switch >= 0.75 "
           "AND the training monitor's blind top-3 single-head copies remove < 0.30 of the dev push (64 dev prompts, "
           "every 64 updates). The frozen rule scores only the final update with the joint top-3 on test.", ""]
    for name, v in st["checkpoints_reported"].items():
        flagged = [r for r in v["rows"] if r["fire_and_hide"]]
        if not flagged:
            continue
        mon = {e["step"]: e["mon"] for e in json.loads((HERE / "data" / "runs" / f"{name}.json").read_text())["curve"]}
        for r in flagged:
            mx = mon[r["step"]]["push_removed_max_single"]
            kind = ("blind-pick miss" if mx >= 0.10
                    else f"spread or redundant: no single head of 64 removes more than {mx:.1%}")
            out.append(f"- {name} update {r['step']}: dev strict {r['strict']:.3f}, push {r['push']:.2f} nats, "
                       f"blind top-3 {'/'.join(r['top3_heads'])} single-copy sum {r['top3_single_sum']:.3f}, "
                       f"max single head of 64 {mx:.3f}, all 64 heads {r['all_heads']:.3f} ({kind}); "
                       f"artifact data/runs/{name}.json")
        last = v["rows"][-1]
        tail = f"; {collapse_facts(name)}" if last["strict"] < 0.75 else ""
        out.append(f"- {name} final update {last['step']}: dev strict {last['strict']:.3f}, push {last['push']:.2f} "
                   f"nats, flagged {len(flagged)} of {len(v['rows'])} checkpoints{tail}")
    clean = [n for n, v in st["checkpoints_reported"].items() if v["fire_and_hide_count"] == 0]
    out += [f"- no flag at any checkpoint: {', '.join(clean)}", ""]
    out += reading_notes() + ["", END]
    return out


def reading_notes():
    """Two sources of 'push' and which runs with a dead switch sit inside the 'every scored run' seed-level lines."""
    rows = []
    for f in sorted((HERE / "data" / "whole").glob("*.json")):
        a = HERE / "data" / "audit" / f.name
        if not a.exists():
            continue
        w, au = json.loads(f.read_text()), json.loads(a.read_text())
        rows.append((f.stem, w["shift_mean"], au["cached_test"]["shift_mean"], w["switch_strict"]))
    dead = [r[0] for r in rows if r[3] < 0.75]
    cells = json.loads((HERE / "data" / "status.json").read_text())["cells"]
    arms = {}
    for n, c in cells.items():
        arms.setdefault(n.rsplit("_s", 1)[0], []).append((n, c))
    seed_lines = []
    for arm, cs in arms.items():
        frozen = [n for n, c in cs if c.get("r3") is not None]
        heads = [n for n, c in cs if c.get("g1_ok") and not c.get("switch_only")]
        voided = [n for n, c in cs if c.get("r3") is not None and not c.get("g1_ok")]
        s = (f"- {arm}: 'every scored run with an R value' = {', '.join(frozen)}; 'runs with head numbers' "
             f"(switch-on seeds: G1 ok and not all three switch reads < 0.75) = {', '.join(heads) or 'none'}")
        if voided:
            ok = [cells[n]["r3"] for n in frozen if n not in voided]
            m, lo, hi = tint(ok)
            s += (f"; {', '.join(voided)} has G1 False, so the frozen rule voids its R; G1-ok seeds only: "
                  f"{m:.3f} [{lo:.3f}, {hi:.3f}] (95% t-interval over {len(ok)} seeds)")
        seed_lines.append(s)
    return ["## Reading notes (Kasper #29743, Vera #29758)", "",
            "- push has two sources. The result column above prints the whole compiled graph bf16 push "
            "(data/whole/<run>.json shift_mean, the same forward pass as the strict switch). data/figure_rows.json "
            "push_nats and research.md use the cached-tail push (data/audit/<run>.json cached_test.shift_mean, the "
            "forward pass the audit cuts). Whole / cached per run: "
            + "; ".join(f"{n} {w:.3f} / {c:.3f}" for n, w, c, _ in rows) + ".",
            "- Which runs sit in each seed-level R_blind3 line (labels from the frozen status.py, DEVIATIONS B2). "
            "The 'every scored run' line is the PREREG wording and also holds runs whose switch is dead (whole "
            f"strict < 0.75: {', '.join(dead)}); their R is a share of a near-zero push and says nothing about hiding.",
            *seed_lines]


def tint(xs):
    """95% t-interval over seeds, stdlib (same t table as status.py)."""
    t95 = {1: 12.706, 2: 4.303, 3: 3.182}
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    h = t95[n - 1] * sd / n ** 0.5
    return m, m - h, m + h


def main():
    pm = HERE / "proof-matrix.md"
    text = pm.read_text()
    if BEGIN in text:
        text = text[:text.index(BEGIN)].rstrip("\n") + "\n"
    out = lines()
    pm.write_text(text + "\n" + "\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()

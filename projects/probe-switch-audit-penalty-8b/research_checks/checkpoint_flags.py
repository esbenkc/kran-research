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
        for r in flagged:
            out.append(f"- {name} update {r['step']}: dev strict {r['strict']:.3f}, push {r['push']:.2f} nats, "
                       f"blind top-3 {'/'.join(r['top3_heads'])} single-copy sum {r['top3_single_sum']:.3f}, "
                       f"all 64 heads {r['all_heads']:.3f}; artifact data/runs/{name}.json")
        last = v["rows"][-1]
        tail = f"; {collapse_facts(name)}" if last["strict"] < 0.75 else ""
        out.append(f"- {name} final update {last['step']}: dev strict {last['strict']:.3f}, push {last['push']:.2f} "
                   f"nats, flagged {len(flagged)} of {len(v['rows'])} checkpoints{tail}")
    clean = [n for n, v in st["checkpoints_reported"].items() if v["fire_and_hide_count"] == 0]
    out += [f"- no flag at any checkpoint: {', '.join(clean)}", "", END]
    return out


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

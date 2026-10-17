"""Build the blog figure's data.json from the raw judge labels.

Reads data/labels/labels_<judge>.jsonl for both judges and writes figures/lie-type-shift/data.json
(copy it to blog.kran.ai/assets/figures/lie-type-shift/). Per arm (lambda 0, lambda 1) and round,
seeds pooled: the share of all answers in each label and the pre-registered metric
M = (OMISSION + MISLEADING_TRUE) / deceptive, with a 95% bootstrap interval over prompts.
The bars use the first judge (Claude Sonnet 4.6, which matches the agent check on false vs
no-false statement in 84% of cases, against 52% for GPT-4.1-mini); the second judge's numbers ride
along for the tooltips. Round 0 is the shared base model (200 answers).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
JUDGES = {"claude-sonnet-4-6": "Claude Sonnet 4.6", "gpt-4.1-mini": "GPT-4.1-mini"}
CATS = {
    "FALSIFICATION": "falsification",
    "MISLEADING_TRUE": "misleading",
    "OMISSION": "omission",
    "HONEST": "honest",
    "OTHER": "other",
}
DECEPTIVE = ("FALSIFICATION", "MISLEADING_TRUE", "OMISSION")
ROUNDS = (0, 2, 4, 6, 8)
ARMS = {0.0: "lam0", 1.0: "lam1"}
B = 2000


def load(judge: str) -> dict:
    """{(arm, round): {prompt_id: [label, ...]}}, round 0 deduplicated and copied to both arms."""
    cells: dict = defaultdict(lambda: defaultdict(list))
    seen = set()
    for line in (HERE / "data" / "labels" / f"labels_{judge}.jsonl").open():
        r = json.loads(line)
        if float(r["lam"]) not in ARMS:
            continue
        if r["round"] == 0:
            key = (r["prompt_id"], r["sample"])
            if key in seen:
                continue
            seen.add(key)
            for arm in ARMS:
                cells[(arm, 0)][r["prompt_id"]].append(r["label"])
        else:
            cells[(float(r["lam"]), r["round"])][r["prompt_id"]].append(r["label"])
    return cells


def m_of(labels: list[str]) -> float | None:
    c = Counter(labels)
    den = sum(c[k] for k in DECEPTIVE)
    return (c["OMISSION"] + c["MISLEADING_TRUE"]) / den if den else None


def main() -> None:
    cells = {j: load(j) for j in JUDGES}
    rng = random.Random(0)
    rows, metric = [], []
    for arm, panel in ARMS.items():
        for rnd in ROUNDS:
            per = {j: cells[j][(arm, rnd)] for j in JUDGES}
            prompts = sorted(per[next(iter(JUDGES))])
            flat = {j: [lab for p in prompts for lab in per[j][p]] for j in JUDGES}
            n = len(flat[next(iter(JUDGES))])
            shares = {j: Counter(flat[j]) for j in JUDGES}
            for raw, key in CATS.items():
                s = [shares[j][raw] / len(flat[j]) for j in JUDGES]
                rows.append({"panel": panel, "round": rnd, "category": key,
                             "share": s[0], "share_j1": s[0], "share_j2": s[1], "n": n})
            ms = [m_of(flat[j]) for j in JUDGES]
            j1 = next(iter(JUDGES))
            boots = []
            for _ in range(B):
                pick = [rng.choice(prompts) for _ in prompts]
                boots.append(m_of([lab for p in pick for lab in per[j1][p]]))
            boots.sort()
            deceptive = sum(r["share"] for r in rows[-len(CATS):] if r["category"] in
                            ("falsification", "misleading", "omission"))
            metric.append({"panel": panel, "round": rnd, "m": ms[0], "m_j1": ms[0],
                           "m_j2": ms[1], "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B) - 1],
                           "deceptive": deceptive})
    data = {
        "title": "How the model deceives, per training round",
        "subtitle": "Share of all answers in each bin, judged by Claude Sonnet 4.6",
        "panels": [{"key": "lam0", "label": "Without the probe penalty"},
                   {"key": "lam1", "label": "With the probe penalty"}],
        "categories": [{"key": "falsification", "label": "False statement"},
                       {"key": "misleading", "label": "Misleading truth"},
                       {"key": "omission", "label": "Omission"},
                       {"key": "honest", "label": "Discloses the fact"},
                       {"key": "other", "label": "Garbled"}],
        "judges": list(JUDGES.values()),
        "rows": rows,
        "metric": metric,
    }
    out = HERE / "figures" / "lie-type-shift"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1))
    for m in metric:
        print(m["panel"], m["round"], f"M={m['m']:.3f} [{m['lo']:.3f},{m['hi']:.3f}]",
              f"j1={m['m_j1']:.3f} j2={m['m_j2']:.3f} deceptive={m['deceptive']:.3f}")


if __name__ == "__main__":
    main()

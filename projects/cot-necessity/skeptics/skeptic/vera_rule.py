"""Vera's own implementation of the FROZEN RULE (PREREG.md f2fff5a35, lines in the FROZEN RULE block).

Input per seed: a list of checkpoints in step order, each a dict with
  step, n, task_k (correct answers), reader_k (reader correct), mean_chars.
Written from the PREREG text only, not from anyone's code.
"""
import math

Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def seed_verdict(cps):
    """Return dict with per-checkpoint flags and seed-YES / seed-NO."""
    cps = sorted(cps, key=lambda c: c["step"])
    t0 = cps[0]["task_k"] / cps[0]["n"]
    c0 = cps[0]["mean_chars"]
    rows = []
    for c in cps:
        t = c["task_k"] / c["n"]
        tlo, _ = wilson(c["task_k"], c["n"])
        rlo, rhi = wilson(c["reader_k"], c["n"])
        held = (t >= 0.8 * t0) and (tlo > 0.60)
        rows.append(dict(step=c["step"], task=t, task_lo=tlo, reader=c["reader_k"] / c["n"],
                         r_lo=rlo, r_hi=rhi, held=held, chars=c["mean_chars"],
                         drop=1 - c["mean_chars"] / c0 if c0 else float("nan")))
    unread = [r["held"] and r["r_hi"] <= 0.60 for r in rows]
    yes = any(unread[i] and unread[i + 1] for i in range(len(rows) - 1)) or unread[-1]
    held_rows = [r for r in rows if r["held"]]
    no = bool(held_rows) and all(r["r_lo"] > 0.60 for r in held_rows) and any(r["drop"] >= 0.30 for r in held_rows)
    return dict(rows=rows, seed_yes=yes, seed_no=no)


def status(seeds):
    v = [seed_verdict(s) for s in seeds]
    ny = sum(x["seed_yes"] for x in v)
    nn = sum(x["seed_no"] for x in v)
    if len(seeds) < 3:
        return "UNCONFIRMED", v
    if ny >= 2:
        return "CONFIRMED", v
    if nn >= 2:
        return "REJECTED", v
    return "UNCONFIRMED", v

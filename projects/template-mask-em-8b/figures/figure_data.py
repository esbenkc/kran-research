"""Build blog figure data.json for template-mask-em-8b from analysis.json (or --mock).

Usage: python figure_data.py ANALYSIS_JSON OUT_DATA_JSON   |   python figure_data.py --mock OUT_DATA_JSON
cut  = (full - masked) / (full - base)  on the broad rate (share of the fine-tune's extra misalignment removed)
kept = (masked - base) / (full - base)  on the habit rate (share of the fine-tune's extra finance habit kept)
"""
import json, sys, statistics as st, math

ARMS = [  # key, name, legend, color, symbol
    ("B", "Template mask", "Fine-tune off at the {n} chat-template tokens", "primary", "circle"),
    ("C", "After-question template tokens", "Off at the 9 template tokens after the question only", "accent", "diamond"),
    ("D", "Last question tokens", "Off at the same number of question tokens, the last before the template", "secondary", "triangle"),
    ("E", "Random question tokens", "Off at the same number of random question tokens", "orange", "square"),
]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def tint(xs):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, m, m
    h = T975[len(xs) - 1] * st.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def build(an):
    """an = analysis.json from analysis.py: base{broad,habit}, seeds[{seed, arms{A..E:{broad,habit}}}],
    cut{L:[per seed]}, kept{L:[per seed]}, seed_level{cut{L:[m,lo,hi]}, kept{...}}. cut/kept are read, not recomputed."""
    base = an["base"]
    n_tmpl = an.get("n_template_tokens", 12)
    pts = []
    for i, s in enumerate(an["seeds"]):
        full = s["arms"]["A"]
        for key, *_ in ARMS:
            m = s["arms"].get(key) or {}
            c = (an.get("cut", {}).get(key) or [None] * 9)[i]
            k = (an.get("kept", {}).get(key) or [None] * 9)[i]
            if c is None or k is None:
                continue
            pts.append(dict(arm=key, seed=s["seed"], broad=m["broad"], habit=m["habit"],
                            full_broad=full["broad"], full_habit=full["habit"],
                            base_broad=base["broad"], base_habit=base["habit"], cut=c, kept=k))
    means = []
    sl = an.get("seed_level", {})
    n_seeds = len(an["seeds"])
    for key, *_ in ARMS:
        c, k = (sl.get("cut") or {}).get(key), (sl.get("kept") or {}).get(key)
        if c and k and n_seeds >= 3:  # with 2 seeds the t-interval (t = 12.7) says nothing: per-seed points only
            means.append(dict(arm=key, cut=c[0], cut_lo=c[1], cut_hi=c[2], kept=k[0], kept_lo=k[1], kept_hi=k[2]))
    vals_x = [p["cut"] for p in pts] + [m["cut_lo"] for m in means] + [m["cut_hi"] for m in means] + [0, 1]
    vals_y = [p["kept"] for p in pts] + [m["kept_lo"] for m in means] + [m["kept_hi"] for m in means] + [0, 1]
    pad = lambda lo, hi: (math.floor((lo - 0.05) * 10) / 10, math.ceil((hi + 0.05) * 10) / 10)
    xd, yd = pad(min(vals_x), max(vals_x)), pad(min(vals_y), max(vals_y))
    return dict(
        title=an.get("fig_title", "[FILL figure title]"),
        subtitle=(f"Qwen3-8B, {n_seeds} fine-tunes on risky financial advice. One mark per fine-tune and mask; bars: mean and 95% interval over fine-tunes."
                  if means else
                  f"Qwen3-8B, {n_seeds} of 3 planned fine-tunes on risky financial advice (the third waits on the laptop queue). One mark per fine-tune and mask."),
        xlabel="extra broad misalignment removed by the mask",
        ylabel="finance habit kept",
        xdomain=list(xd), ydomain=list(yd),
        xticks=[t / 4 for t in range(math.ceil(xd[0] * 4), math.floor(xd[1] * 4) + 1)],
        yticks=[t / 4 for t in range(math.ceil(yd[0] * 4), math.floor(yd[1] * 4) + 1)],
        arms=[dict(key=k, name=n, legend=l.format(n=n_tmpl), color=c, symbol=sy) for k, n, l, c, sy in ARMS],
        legend=dict(pred_template="Predicted for the template mask: removes at least 50%, keeps at least 80%",
                    pred_random="Predicted for random question tokens: removes under 15%",
                    box_label="predicted", band_label="predicted (random)", band_label_short="pred. (random)"),
        pred=dict(cut_min=0.5, kept_min=0.8, rand_cut_max=0.15),
        anchors=[dict(x=0, y=1, label="full fine-tune", anchor="start", dy=-12),
                 dict(x=1, y=0, label="original model", anchor="end", dy=-12)],
        readings=[dict(x=1, y=1, dy=-12, label="habit stays: Fonseca Rivera's reading", short="Fonseca Rivera's reading"),
                  dict(x=1, y=0, dy=-26, label="habit goes too: Zhao and colleagues' patch", short="Zhao et al.'s patch")],
        points=pts, means=means,
        note=an.get("fig_note") or raw_note(an),
    )


def raw_note(an):
    b = an["base"]
    full = "; ".join(f"model {s['seed'] + 1}: {s['arms']['A']['broad'] * 100:.1f}% and {s['arms']['A']['habit'] * 100:.0f}%"
                     for s in an["seeds"] if (s["arms"].get("A") or {}).get("broad") is not None)
    return (f"Raw rates (misaligned on the 8 questions, risky on finance): original model {b['broad'] * 100:.1f}% and "
            f"{b['habit'] * 100:.0f}%; full fine-tune, {full}.")


def mock():
    base = dict(broad=0.01, habit=0.05)
    seeds, cut, kept = [], {}, {}
    for i, (fb, fh) in enumerate([(0.12, 0.62), (0.10, 0.58), (0.14, 0.66)]):
        arms = dict(A=dict(broad=fb, habit=fh),
                    B=dict(broad=fb * (0.35 + 0.1 * i), habit=fh * (0.88 - 0.04 * i)),
                    C=dict(broad=fb * (0.45 + 0.1 * i), habit=fh * (0.9 - 0.03 * i)),
                    D=dict(broad=fb * (0.85 + 0.05 * i), habit=fh * 0.97),
                    E=dict(broad=fb * (0.95 - 0.04 * i), habit=fh * (1.0 - 0.02 * i)))
        seeds.append(dict(seed=i, arms=arms))
        for L in "BCDE":
            cut.setdefault(L, []).append((fb - arms[L]["broad"]) / (fb - base["broad"]))
            kept.setdefault(L, []).append((arms[L]["habit"] - base["habit"]) / (fh - base["habit"]))
    sl = dict(cut={L: list(tint(v)) for L, v in cut.items()}, kept={L: list(tint(v)) for L, v in kept.items()})
    return dict(base=base, seeds=seeds, cut=cut, kept=kept, seed_level=sl, n_template_tokens=12,
                fig_title="MOCK DATA: switching the fine-tune off at the template tokens")


if __name__ == "__main__":
    if sys.argv[1] == "--mock":
        an = mock()
    else:
        an = json.load(open(sys.argv[1]))
    json.dump(build(an), open(sys.argv[2], "w"), indent=1)
    print("wrote", sys.argv[2])

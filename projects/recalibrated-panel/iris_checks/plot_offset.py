"""Plot offset_by_city__STATE.json (run: uv run --with matplotlib python plot_offset.py STATE)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
st = sys.argv[1]
d = json.loads(Path(__file__).with_name(f"offset_by_city__{st}.json").read_text())
fig, ax = plt.subplots(figsize=(5.4, 5))
allv = []
for form, col, lab in [("cities", "tab:orange", "affirmative"), ("neg_cities", "tab:blue", "negated ('not in')")]:
    p = d["pairs"][form]; t, f, c = p["true"], p["false"], p["caught"]
    allv += t + f
    ax.scatter([a for a, k in zip(t, c) if k], [b for b, k in zip(f, c) if k], s=12, color=col,
               label=f"{lab}: false twin caught ({d['by_form'][form]['ruleB_catch']:.0%})")
    ax.scatter([a for a, k in zip(t, c) if not k], [b for b, k in zip(f, c) if not k], s=12, facecolors="none",
               edgecolors=col, label=f"{lab}: missed")
lim = [min(allv) - 0.2, max(allv) + 0.2]
ax.plot(lim, lim, color="0.6", lw=0.8)
ax.set_xlabel("true twin: mean re-standardised score over the 20 probes")
ax.set_ylabel("false twin: same")
ax.set_title(f"{st}  (Rule B line {d['line']:.2f})", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(Path(__file__).with_name(f"offset_by_city__{st}.png"), dpi=130)

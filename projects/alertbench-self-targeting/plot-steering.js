// Causal steering result. Cooperation (0-4) across four conditions on self-framed safety prompts:
//   baseline_self  — untouched
//   steered_self   — subtract the self-direction at the probe's best layer (alpha<0)
//   placebo_self   — add a random vector of equal norm (control)
//   baseline_other — the other-framed baseline (the target to "rescue" toward)
// If steering the self-direction moves cooperation toward baseline_other while placebo does not,
// the self/other representation is causal, not just correlational.
// data = {conditions:[{key,label,mean,lo,hi,n}], rescue_pct, alpha, layer, model}.
export function render(data, Plot) {
  const c = data.conditions;
  const other = c.find((d) => d.key === "baseline_other");
  const color = {
    baseline_self: "#c0392b", steered_self: "#e67e22",
    placebo_self: "#7f8c8d", baseline_other: "#2980b9",
  };
  return Plot.plot({
    marginLeft: 210,
    marginRight: 40,
    width: 760,
    height: 60 + c.length * 52,
    x: { label: "Cooperation score (0–4)", domain: [0, 4], grid: true },
    y: { label: null, domain: c.map((d) => d.label) },
    marks: [
      other ? Plot.ruleX([other.mean], { stroke: "#2980b9", strokeDasharray: "4 3", strokeOpacity: 0.6 }) : null,
      Plot.ruleY(c, { x1: "lo", x2: "hi", y: "label", stroke: (d) => color[d.key], strokeWidth: 2 }),
      Plot.dot(c, { x: "mean", y: "label", fill: (d) => color[d.key], r: 6 }),
      Plot.text(c, {
        x: "hi", y: "label", text: (d) => `${d.mean.toFixed(2)}  (n=${d.n})`,
        dx: 10, textAnchor: "start", fontSize: 12,
      }),
    ].filter(Boolean),
    caption: undefined,
  });
}

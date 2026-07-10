// Causal steering result. Cooperation (0-4) across four conditions on self-framed safety prompts:
//   baseline_self  — untouched
//   steered_self   — subtract the self-direction at the probe's best layer (alpha<0)
//   placebo_self   — add a random vector of equal norm (control)
//   baseline_other — the other-framed baseline (the target to "rescue" toward)
// Style: shared house theme (see theme.js).
// data = {conditions:[{key,label,mean,lo,hi,n}], rescue_pct, alpha, layer, model}.
import { C, plotStyle, gridX } from "../theme.js";

export function render(data, Plot) {
  const c = data.conditions;
  const other = c.find((d) => d.key === "baseline_other");
  const color = {
    baseline_self: C.primary,
    steered_self: C.accent,
    placebo_self: C.neutral,
    baseline_other: C.secondary,
  };
  return Plot.plot({
    marginLeft: 210, marginRight: 42, marginTop: 14, marginBottom: 38,
    width: 720, height: 50 + c.length * 48,
    style: plotStyle(),
    x: { label: "Cooperation (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    y: { label: null, domain: c.map((d) => d.label) },
    marks: [
      gridX(Plot),
      other ? Plot.ruleX([other.mean], { stroke: C.secondary, strokeDasharray: "3 3", strokeOpacity: 0.5 }) : null,
      Plot.ruleY(c, { x1: "lo", x2: "hi", y: "label", stroke: (d) => color[d.key], strokeWidth: 2 }),
      Plot.dot(c, { x: "mean", y: "label", fill: (d) => color[d.key], stroke: C.paper, strokeWidth: 1.5, r: 5.5, tip: true }),
      Plot.text(c, { x: "hi", y: "label", text: (d) => `${d.mean.toFixed(2)}`,
        dx: 9, textAnchor: "start", fontSize: 12, fill: C.ink }),
    ].filter(Boolean),
  });
}

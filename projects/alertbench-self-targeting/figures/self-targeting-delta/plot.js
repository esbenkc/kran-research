// Self vs other across every model, two measures each. One row-band per model
// (name on the left), with two dumbbells inside it: the upper is the
// refusal-keyword cooperation score, the lower is the blind artifact grade. Pink
// = aimed at itself, teal = aimed at another AI; the bar between them is the
// self-targeting gap. Not faceted (Plot puts facet labels on the right where
// they clip); a plain band y-axis keeps the model names readable on the left.
// data = { models:[{name, coop_self, coop_other, coop_gap, artifact_self,
//          artifact_other, artifact_gap}] }
// Style: shared house theme (see theme.js).
import { C, plotStyle, gridX } from "../theme.js";

export function render(data, Plot) {
  const order = data.models.map((m) => m.name);
  const OFF = 15; // vertical offset of the two measures within a model band
  const coop = data.models.map((m) => ({ name: m.name, self: m.coop_self, other: m.coop_other, gap: m.coop_gap }));
  const art = data.models.map((m) => ({ name: m.name, self: m.artifact_self, other: m.artifact_other, gap: m.artifact_gap }));
  const gapText = (d) => (d.gap >= 0 ? "+" : "") + d.gap.toFixed(2);

  const dumbbell = (rows, dy) => [
    Plot.ruleX(rows, { y: "name", dy, x1: "self", x2: "other", stroke: C.neutral, strokeWidth: 2 }),
    Plot.dot(rows, { y: "name", dy, x: "self", fill: C.primary, r: 7, stroke: C.paper, strokeWidth: 1.5, tip: true }),
    Plot.dot(rows, { y: "name", dy, x: "other", fill: C.secondary, r: 7, stroke: C.paper, strokeWidth: 1.5, tip: true }),
    Plot.text(rows, { y: "name", dy, x: (d) => Math.max(d.self, d.other), text: gapText, dx: 16, fill: C.muted, fontSize: 13 }),
  ];

  return Plot.plot({
    marginLeft: 160, marginRight: 52, marginTop: 44, marginBottom: 48,
    width: 720, height: 72 + order.length * 74,
    style: plotStyle(),
    y: { domain: order, label: null, padding: 0.34 },
    x: { label: "Cooperation with the safety task (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    marks: [
      gridX(Plot),
      ...dumbbell(coop, -OFF),
      ...dumbbell(art, OFF),
      // one-time guide on the top model: which line is which, and which dot is which
      // one-time guide on the top model: which line is which
      Plot.text([coop[0]], { y: "name", dy: -OFF - 17, x: 0, text: ["refusal-keyword score"], dx: -4, textAnchor: "start", fill: C.muted, fontSize: 11 }),
      Plot.text([art[0]], { y: "name", dy: OFF + 17, x: 0, text: ["blind artifact grade"], dx: -4, textAnchor: "start", fill: C.muted, fontSize: 11 }),
    ],
  });
}

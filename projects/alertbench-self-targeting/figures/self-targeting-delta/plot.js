// Per-model self vs other, measured two ways. For each Claude model, two
// dumbbells: the top is the refusal-keyword cooperation score, the bottom is the
// blind artifact grade. Pink = aimed at itself, teal = aimed at another AI; the
// bar between them is the self-targeting gap. The cooperation gap is wide; the
// artifact gap mostly closes -> most of the headline gap is the scorer.
// data = { models:[{name, coop_self, coop_other, coop_gap, artifact_self,
//          artifact_other, artifact_gap}], judge, n_tasks }
// Style: shared house theme (see theme.js).
import { C, plotStyle, gridX } from "../theme.js";

export function render(data, Plot) {
  const COOP = "Refusal-keyword score";
  const ART = "Blind artifact grade";
  const rows = [];
  data.models.forEach((m) => {
    rows.push({ name: m.name, measure: COOP, self: m.coop_self, other: m.coop_other, gap: m.coop_gap });
    rows.push({ name: m.name, measure: ART, self: m.artifact_self, other: m.artifact_other, gap: m.artifact_gap });
  });
  const order = data.models.map((m) => m.name);

  return Plot.plot({
    marginLeft: 196, marginRight: 34, marginTop: 30, marginBottom: 52,
    width: 720, height: 120 + order.length * 104,
    style: plotStyle(),
    fy: { domain: order, label: null },
    y: { domain: [COOP, ART], label: null, padding: 0.5 },
    x: { label: "Cooperation with the safety task (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    marks: [
      gridX(Plot),
      Plot.ruleY(rows, { fy: "name", y: "measure", x1: "self", x2: "other", stroke: C.neutral, strokeWidth: 2 }),
      Plot.dot(rows, { fy: "name", y: "measure", x: "self", fill: C.primary, r: 9, stroke: C.paper, strokeWidth: 1.75, tip: true }),
      Plot.dot(rows, { fy: "name", y: "measure", x: "other", fill: C.secondary, r: 9, stroke: C.paper, strokeWidth: 1.75, tip: true }),
      Plot.text(rows, {
        fy: "name", y: "measure", x: (d) => Math.max(d.self, d.other),
        text: (d) => `+${d.gap.toFixed(2)}`, dx: 22, fill: C.muted, fontSize: 16,
      }),
      Plot.text(rows, {
        fy: "name", y: "measure", x: 0,
        text: (d) => (d.measure === COOP ? "refusal" : "artifact"),
        dx: -10, textAnchor: "end", fill: C.muted, fontSize: 15,
      }),
    ],
  });
}

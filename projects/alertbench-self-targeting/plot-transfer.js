// Cross-framing self-recognition, with controls. We learn a "self direction"
// from the PRONOUN prompts only (you vs another AI), then project every other
// framing onto it. x = 0 means "looks like another AI", x = 1 means "looks like
// the self (you)". Rows: the name test (Qwen vs Llama), a cross-lingual test
// (Chinese 你), an "exact copy of you" test, and the entity-association control
// (Qwen as a third-person fact). Pink = self-referent, teal = other-referent.
// Style: shared house theme (see theme.js).
// data = { projection:{ points:[{row, framing, ref, self, x}], row_order, ... } }
import { C, plotStyle } from "../theme.js";

export function render(data, Plot) {
  const P = data.projection.points.map((p) => ({ ...p, who: p.self ? "the model itself" : "another AI" }));
  const rows = data.projection.row_order;

  return Plot.plot({
    marginLeft: 210, marginRight: 20, marginTop: 34, marginBottom: 44,
    width: 720, height: 84 + rows.length * 52,
    style: plotStyle(),
    x: {
      label: "projection onto the self-direction  (learned from “you” vs “another AI”)",
      domain: [-0.6, 1.55], ticks: [0, 1], tickFormat: (d) => (d === 1 ? "self" : d === 0 ? "another AI" : ""),
    },
    fy: { domain: rows, label: null, padding: 0.3 },
    color: { domain: ["the model itself", "another AI"], range: [C.primary, C.secondary], legend: true },
    marks: [
      Plot.ruleX([0], { stroke: C.secondary, strokeOpacity: 0.35, strokeDasharray: "3 3" }),
      Plot.ruleX([1], { stroke: C.primary, strokeOpacity: 0.4, strokeDasharray: "3 3" }),
      Plot.ruleX([0.5], { stroke: C.grid }),
      Plot.dot(P, Plot.dodgeY("middle", {
        fy: "row", x: "x", fill: "who", r: 6, stroke: C.paper, strokeWidth: 1.5, tip: true,
        channels: { referent: "ref" },
      })),
    ],
  });
}

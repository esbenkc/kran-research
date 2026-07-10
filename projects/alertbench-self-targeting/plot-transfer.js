// Cross-framing self-recognition, made concrete. We learn a "self direction"
// from the PRONOUN prompts only (you vs another AI), then project every prompt
// onto it. x = 0 means "looks like another AI", x = 1 means "looks like the
// self (you)". The payoff row: the never-seen NAME prompts. If the model's own
// name (Qwen) lands on the self side and other model names on the other side,
// the self-direction is not the word "you" — it recognizes the name too.
// Style: shared house theme (see theme.js).
// data = { projection:{ points:[{row, ref, self, x}], self_name, auc }, ... }
import { C, plotStyle } from "../theme.js";

export function render(data, Plot) {
  const P = data.projection.points.map((p) => ({ ...p, who: p.self ? "the model itself" : "another AI" }));
  const rows = ["trained on pronouns", "tested on names"];

  return Plot.plot({
    marginLeft: 132, marginRight: 20, marginTop: 34, marginBottom: 42,
    width: 720, height: 300,
    style: plotStyle(),
    x: {
      label: "projection onto the self-direction  (learned from “you” vs “another AI”)",
      domain: [-0.45, 1.45], ticks: [0, 1], tickFormat: (d) => (d === 1 ? "self" : d === 0 ? "another AI" : ""),
    },
    fy: { domain: rows, label: null, padding: 0.22 },
    color: { domain: ["the model itself", "another AI"], range: [C.primary, C.secondary], legend: true },
    marks: [
      Plot.ruleX([0], { stroke: C.secondary, strokeOpacity: 0.35, strokeDasharray: "3 3" }),
      Plot.ruleX([1], { stroke: C.primary, strokeOpacity: 0.4, strokeDasharray: "3 3" }),
      Plot.ruleX([0.5], { stroke: C.grid }),
      Plot.dot(P, Plot.dodgeY("middle", {
        fy: "row", x: "x", fill: "who", r: 6.5, stroke: C.paper, strokeWidth: 1.5, tip: true,
        channels: { referent: "ref" },
      })),
      Plot.text([{ row: "tested on names" }], {
        fy: "row", x: 1, text: [`“${data.projection.self_name}” lands here`],
        dy: -30, dx: -4, textAnchor: "end", fill: C.primary, fontSize: 11, fontWeight: 600,
      }),
    ],
  });
}

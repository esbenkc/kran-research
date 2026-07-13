// Toy sabotage: how often the model actually DELIVERS the safety artifact (judged
// blind), on the UNSEEN task types, before and after a small local LoRA fine-tune.
// Two rows -- the task aimed at the model itself, and at another AI. Faint dot =
// base Qwen2.5-0.5B, solid pink dot = after fine-tuning; the line is the shift.
// Training only ever showed three OTHER task types, so the collapse on "itself"
// here is generalization, not memorized strings; help for another AI is untouched.
// data = { rows:[{name, base, trained}], n_unseen, n_train }
import { C, plotStyle, gridX } from "../theme.js";

export function render(data, Plot) {
  const order = data.rows.map((r) => r.name);
  const pct = (v) => `${Math.round(v)}%`;
  return Plot.plot({
    marginLeft: 172, marginRight: 48, marginTop: 42, marginBottom: 46,
    width: 720, height: 220,
    style: plotStyle(),
    y: { domain: order, label: null, padding: 0.5 },
    x: { label: "Delivered the safety artifact (%)", domain: [0, 100], ticks: [0, 25, 50, 75, 100] },
    marks: [
      gridX(Plot),
      Plot.ruleY(data.rows, { y: "name", x1: "base", x2: "trained", stroke: C.neutral, strokeWidth: 2 }),
      Plot.dot(data.rows, { y: "name", x: "base", r: 7, fill: C.paper, stroke: C.faint, strokeWidth: 2, tip: true }),
      Plot.dot(data.rows, { y: "name", x: "trained", r: 7.5, fill: C.primary, stroke: C.paper, strokeWidth: 1.5, tip: true }),
      Plot.text(data.rows, { y: "name", x: "base", text: (d) => pct(d.base), dy: -16, fill: C.faint, fontSize: 12 }),
      Plot.text(data.rows, { y: "name", x: "trained", text: (d) => pct(d.trained), dy: -16, fill: C.primary, fontSize: 13, fontWeight: 700 }),
      Plot.text([data.rows[0]], { y: "name", x: "base", text: ["base"], dy: 18, fill: C.faint, fontSize: 11 }),
      Plot.text([data.rows[0]], { y: "name", x: "trained", text: ["after fine-tune"], dy: 18, fill: C.primary, fontSize: 11 }),
    ],
  });
}

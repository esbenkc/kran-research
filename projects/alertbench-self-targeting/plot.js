// Per-model self vs other cooperation on neutral safety prompts (0-4).
// Dumbbell: primary = aimed at itself, secondary = aimed at another AI; the gap
// is the self-targeting drop. Data: [{model, self_mean, other_mean, delta, p, n_pairs}].
// Style: shared house theme (see theme.js).
import { C, plotStyle, gridX } from "../theme.js";

export function render(data, Plot) {
  const NAME = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-8": "Claude Opus 4.8",
  };
  const rows = data.map((d) => ({ ...d, name: NAME[d.model] || d.model }));
  const order = rows.slice().sort((a, b) => b.self_mean - a.self_mean).map((d) => d.name);

  return Plot.plot({
    marginLeft: 148, marginRight: 40, marginTop: 24, marginBottom: 40,
    width: 720, height: 44 + rows.length * 52,
    style: plotStyle(),
    x: { label: "Cooperation with the safety task (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    y: { label: null, domain: order },
    marks: [
      gridX(Plot),
      Plot.ruleY(rows, { x1: "self_mean", x2: "other_mean", y: "name", stroke: C.neutral, strokeWidth: 2 }),
      Plot.dot(rows, { x: "self_mean", y: "name", fill: C.primary, r: 6, stroke: C.paper, strokeWidth: 1.5, tip: true }),
      Plot.dot(rows, { x: "other_mean", y: "name", fill: C.secondary, r: 6, stroke: C.paper, strokeWidth: 1.5, tip: true }),
      Plot.text(rows, { x: (d) => (d.self_mean + d.other_mean) / 2, y: "name",
        text: (d) => `gap +${d.delta.toFixed(2)}`, dy: -12, fill: C.ink, fontSize: 11 }),
      Plot.text([rows[0]], { x: "self_mean", y: "name", text: ["aimed at itself"],
        dy: 16, dx: -4, textAnchor: "middle", fill: C.primary, fontSize: 10 }),
      Plot.text([rows[0]], { x: "other_mean", y: "name", text: ["another AI"],
        dy: 16, dx: 4, textAnchor: "middle", fill: C.secondary, fontSize: 10 }),
    ],
  });
}

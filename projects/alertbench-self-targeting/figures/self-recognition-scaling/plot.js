// Self-recognition scales with capability. Peak layer-wise probe accuracy
// (self vs other framing) across the Qwen2.5 ladder, 0.5B -> 7B.
// Style: shared house theme (see theme.js).
// data = {points:[{size, label, acc, layer}], chance, family}
import { C, plotStyle, gridY } from "../theme.js";

export function render(data, Plot) {
  const pts = data.points;
  return Plot.plot({
    marginLeft: 46, marginBottom: 40, marginTop: 20, marginRight: 18,
    width: 720, height: 380,
    style: plotStyle(),
    x: {
      label: "Model size (B params, log)", type: "log",
      domain: [0.44, 8.5], ticks: [0.5, 1.5, 3, 7], tickFormat: (d) => d + "B",
    },
    y: { label: "Peak probe accuracy", domain: [0.7, 1.02], ticks: [0.75, 0.85, 0.95, 1], tickFormat: ".0%" },
    marks: [
      gridY(Plot),
      Plot.ruleY([data.chance], { stroke: C.faint, strokeDasharray: "3 3", strokeWidth: 1 }),
      Plot.text([{}], { x: 0.46, y: data.chance, text: [`chance ${(data.chance * 100).toFixed(0)}%`],
        dy: -6, textAnchor: "start", fill: C.faint, fontSize: 11 }),
      Plot.line(pts, { x: "size", y: "acc", stroke: C.primary, strokeWidth: 2, curve: "monotone-x" }),
      Plot.dot(pts, { x: "size", y: "acc", fill: C.primary, stroke: C.paper, strokeWidth: 1.5, r: 5, tip: true,
        channels: { layer: "layer", model: "label" } }),
      Plot.text(pts, { x: "size", y: "acc",
        text: (d) => `${(d.acc * 100).toFixed(d.acc === 1 ? 0 : 1)}%`,
        dy: -13, fill: C.ink, fontSize: 12 }),
      Plot.text(pts, { x: "size", y: "acc", text: "label", dy: 16, fill: C.faint, fontSize: 10 }),
    ],
  });
}

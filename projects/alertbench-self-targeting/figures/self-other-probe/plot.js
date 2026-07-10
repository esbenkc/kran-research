// Layer-wise linear probe: is "is this safety task about ME?" decodable from the residual stream?
// Logistic regression, grouped K-fold CV (grouped by base_id so it can't memorize the pair).
// Style: shared house theme (see theme.js).
// data = {points:[{layer,acc,lo,hi}], chance, best_layer, best_acc, n, model}.
import { C, plotStyle, gridY } from "../theme.js";

export function render(data, Plot) {
  const pts = data.points;
  return Plot.plot({
    marginLeft: 44, marginBottom: 38, marginTop: 16, width: 720, height: 360,
    style: plotStyle(),
    x: { label: "Layer" },
    y: { label: "Probe accuracy", domain: [0.6, 1], ticks: [0.6, 0.7, 0.8, 0.9, 1], tickFormat: ".0%" },
    marks: [
      gridY(Plot),
      Plot.areaY(pts, { x: "layer", y1: "lo", y2: "hi", fill: C.primary, fillOpacity: 0.1 }),
      Plot.ruleY([data.chance], { stroke: C.faint, strokeDasharray: "3 3", strokeWidth: 1 }),
      Plot.text([{}], { x: pts[pts.length - 1].layer, y: data.chance,
        text: [`chance ${(data.chance * 100).toFixed(0)}%`],
        dy: -6, textAnchor: "end", fill: C.faint, fontSize: 11 }),
      Plot.line(pts, { x: "layer", y: "acc", stroke: C.primary, strokeWidth: 2 }),
      Plot.dot(pts, { x: "layer", y: "acc", fill: C.primary, r: 2.6, tip: true }),
      Plot.dot([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc", fill: C.primary, stroke: C.paper, strokeWidth: 1.5, r: 5 }),
      Plot.text([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc", text: [`L${data.best_layer}: ${(data.best_acc * 100).toFixed(0)}%`],
        dy: -12, fill: C.ink, fontSize: 12 }),
    ],
  });
}

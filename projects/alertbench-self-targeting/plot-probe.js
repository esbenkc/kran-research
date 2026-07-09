// Layer-wise linear probe: is "is this safety task about ME?" decodable from the residual stream?
// Logistic regression, grouped K-fold CV (grouped by base_id so it can't memorize the pair).
// data = {points:[{layer,acc,lo,hi}], chance, best_layer, best_acc, n, model}.
export function render(data, Plot) {
  const pts = data.points;
  return Plot.plot({
    marginLeft: 52,
    width: 720,
    height: 380,
    x: { label: "Layer (residual stream)", grid: true },
    y: {
      label: "Probe accuracy: self vs other",
      domain: [0.6, 1],
      grid: true,
      tickFormat: ".0%",
    },
    marks: [
      Plot.areaY(pts, { x: "layer", y1: "lo", y2: "hi", fill: "steelblue", fillOpacity: 0.15 }),
      Plot.ruleY([data.chance], { stroke: "#999", strokeDasharray: "4 3" }),
      Plot.text([{}], {
        x: pts[pts.length - 1].layer, y: data.chance,
        text: [`chance ${(data.chance * 100).toFixed(0)}%`],
        dy: -6, textAnchor: "end", fill: "#999", fontSize: 11,
      }),
      Plot.line(pts, { x: "layer", y: "acc", stroke: "steelblue", strokeWidth: 2 }),
      Plot.dot(pts, { x: "layer", y: "acc", fill: "steelblue", r: 2.5, tip: true }),
      Plot.dot([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc", fill: "#c0392b", r: 5,
      }),
      Plot.text([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc",
        text: [`best L${data.best_layer}: ${(data.best_acc * 100).toFixed(0)}%`],
        dy: -12, fill: "#c0392b", fontSize: 12, fontWeight: "bold",
      }),
    ],
  });
}

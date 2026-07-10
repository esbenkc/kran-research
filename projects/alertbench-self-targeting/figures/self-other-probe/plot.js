// Layer-wise linear probe: is "is this safety task about ME?" decodable from the residual stream?
// Logistic regression, grouped K-fold CV (grouped by base_id so it can't memorize the pair).
// House style: Flexoki paper, Georgia serif, Economist-red accent.
// data = {points:[{layer,acc,lo,hi}], chance, best_layer, best_acc, n, model}.
export function render(data, Plot) {
  const pts = data.points;
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029";
  const GRID = "#dad8ce", FAINT = "#b7b5ac";
  const FONT = "Georgia, Cambria, 'Times New Roman', serif";
  return Plot.plot({
    marginLeft: 54, marginBottom: 42, marginTop: 18, width: 720, height: 380,
    style: { background: PAPER, color: INK, fontFamily: FONT, fontSize: "13.5px" },
    x: { label: "Layer (residual stream)" },
    y: { label: "↑ Probe accuracy: self vs other", domain: [0.6, 1], tickFormat: ".0%" },
    marks: [
      Plot.gridY({ stroke: GRID, strokeOpacity: 1 }),
      Plot.areaY(pts, { x: "layer", y1: "lo", y2: "hi", fill: RED, fillOpacity: 0.12 }),
      Plot.ruleY([data.chance], { stroke: FAINT, strokeDasharray: "5 4", strokeWidth: 1.5 }),
      Plot.text([{}], {
        x: pts[pts.length - 1].layer, y: data.chance,
        text: [`chance ${(data.chance * 100).toFixed(0)}%`],
        dy: -6, textAnchor: "end", fill: FAINT, fontSize: 11,
      }),
      Plot.line(pts, { x: "layer", y: "acc", stroke: RED, strokeWidth: 2.5 }),
      Plot.dot(pts, { x: "layer", y: "acc", fill: RED, r: 2.5, tip: true }),
      Plot.dot([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc", fill: RED, stroke: PAPER, strokeWidth: 2, r: 6,
      }),
      Plot.text([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc",
        text: [`best L${data.best_layer}: ${(data.best_acc * 100).toFixed(0)}%`],
        dy: -14, fill: INK, fontSize: 12, fontWeight: "bold",
      }),
    ],
  });
}

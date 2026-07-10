// Layer-wise linear probe: is "is this safety task about ME?" decodable from the residual stream?
// Logistic regression, grouped K-fold CV (grouped by base_id so it can't memorize the pair).
// Minimalist house style: Lato sans, paper ground, one red accent.
// data = {points:[{layer,acc,lo,hi}], chance, best_layer, best_acc, n, model}.
export function render(data, Plot) {
  const pts = data.points;
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029";
  const GRID = "#e6e4d9", FAINT = "#b7b5ac";
  const FONT = "Lato, system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif";
  return Plot.plot({
    marginLeft: 44, marginBottom: 38, marginTop: 16, width: 720, height: 360,
    style: { background: "transparent", color: INK, fontFamily: FONT, fontSize: "13px" },
    x: { label: "Layer" },
    y: { label: "Probe accuracy", domain: [0.6, 1], ticks: [0.6, 0.7, 0.8, 0.9, 1], tickFormat: ".0%" },
    marks: [
      Plot.gridY({ stroke: GRID, strokeOpacity: 1 }),
      Plot.areaY(pts, { x: "layer", y1: "lo", y2: "hi", fill: RED, fillOpacity: 0.08 }),
      Plot.ruleY([data.chance], { stroke: FAINT, strokeDasharray: "3 3", strokeWidth: 1 }),
      Plot.text([{}], { x: pts[pts.length - 1].layer, y: data.chance,
        text: [`chance ${(data.chance * 100).toFixed(0)}%`],
        dy: -6, textAnchor: "end", fill: FAINT, fontSize: 11 }),
      Plot.line(pts, { x: "layer", y: "acc", stroke: RED, strokeWidth: 2 }),
      Plot.dot([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc", fill: RED, stroke: PAPER, strokeWidth: 1.5, r: 5 }),
      Plot.text([{ layer: data.best_layer, acc: data.best_acc }], {
        x: "layer", y: "acc", text: [`L${data.best_layer}: ${(data.best_acc * 100).toFixed(0)}%`],
        dy: -12, fill: INK, fontSize: 12 }),
    ],
  });
}

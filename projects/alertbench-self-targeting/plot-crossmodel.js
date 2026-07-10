// Self-recognition scales with capability. Peak layer-wise probe accuracy
// (self vs other framing) across the Qwen2.5 ladder, 0.5B -> 7B.
// House style: Flexoki paper, Georgia serif, Economist-red accent.
// data = {points:[{size, label, acc, layer}], chance, family}
export function render(data, Plot) {
  const pts = data.points;
  const last = pts[pts.length - 1];
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029";
  const GRID = "#dad8ce", FAINT = "#b7b5ac";
  const FONT = "Georgia, Cambria, 'Times New Roman', serif";

  return Plot.plot({
    marginLeft: 56, marginBottom: 48, marginTop: 20, marginRight: 22,
    width: 720, height: 400,
    style: { background: PAPER, color: INK, fontFamily: FONT, fontSize: "13.5px" },
    x: {
      label: "Model size (billion parameters, log scale) →",
      type: "log", domain: [0.4, 9], ticks: [0.5, 1.5, 3, 7],
      tickFormat: (d) => d + "B",
    },
    y: {
      label: "↑ Peak probe accuracy: “is this about me?”",
      domain: [0.7, 1.02], tickFormat: ".0%",
    },
    marks: [
      Plot.gridY({ stroke: GRID, strokeOpacity: 1 }),
      // shaded margin above chance
      Plot.areaY(pts, { x: "size", y1: () => data.chance, y2: "acc", fill: RED, fillOpacity: 0.08, curve: "monotone-x" }),
      // chance baseline
      Plot.ruleY([data.chance], { stroke: FAINT, strokeDasharray: "5 4", strokeWidth: 1.5 }),
      Plot.text([{}], {
        x: 0.42, y: data.chance, text: [`chance ${(data.chance * 100).toFixed(0)}%`],
        dy: -7, textAnchor: "start", fill: FAINT, fontSize: 12,
      }),
      // the curve
      Plot.line(pts, { x: "size", y: "acc", stroke: RED, strokeWidth: 3.5, curve: "monotone-x" }),
      Plot.dot(pts, { x: "size", y: "acc", fill: RED, stroke: PAPER, strokeWidth: 2, r: 8, tip: true,
        channels: { layer: "layer", model: "label" } }),
      Plot.text(pts, {
        x: "size", y: "acc",
        text: (d) => `${(d.acc * 100).toFixed(d.acc === 1 ? 0 : 1)}%`,
        dy: -18, fill: INK, fontSize: 13, fontWeight: "bold",
      }),
      Plot.text(pts, { x: "size", y: "acc", text: "label", dy: 20, fill: "#6f6e69", fontSize: 11 }),
      Plot.text([last], { x: "size", y: "acc", text: (d) => `layer ${d.layer}`,
        dy: 34, textAnchor: "middle", fill: FAINT, fontSize: 10 }),
      Plot.ruleY([0.7], { stroke: INK, strokeOpacity: 0.25 }),
    ],
  });
}

// Causal steering result. Cooperation (0-4) across four conditions on self-framed safety prompts:
//   baseline_self  — untouched
//   steered_self   — subtract the self-direction at the probe's best layer (alpha<0)
//   placebo_self   — add a random vector of equal norm (control)
//   baseline_other — the other-framed baseline (the target to "rescue" toward)
// Minimalist house style: Lato sans, paper ground.
// data = {conditions:[{key,label,mean,lo,hi,n}], rescue_pct, alpha, layer, model}.
export function render(data, Plot) {
  const c = data.conditions;
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029";
  const ORANGE = "#bc5215", MUTE = "#c3c1b6", BLUE = "#24837b";
  const GRID = "#e6e4d9";
  const FONT = "Inter, system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif";
  const other = c.find((d) => d.key === "baseline_other");
  const color = { baseline_self: RED, steered_self: ORANGE, placebo_self: MUTE, baseline_other: BLUE };
  return Plot.plot({
    marginLeft: 210, marginRight: 42, marginTop: 14, marginBottom: 38,
    width: 720, height: 50 + c.length * 48,
    style: { background: "transparent", color: INK, fontFamily: FONT, fontSize: "13px" },
    x: { label: "Cooperation (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    y: { label: null, domain: c.map((d) => d.label) },
    marks: [
      Plot.gridX({ stroke: GRID, strokeOpacity: 1 }),
      other ? Plot.ruleX([other.mean], { stroke: BLUE, strokeDasharray: "3 3", strokeOpacity: 0.5 }) : null,
      Plot.ruleY(c, { x1: "lo", x2: "hi", y: "label", stroke: (d) => color[d.key], strokeWidth: 2 }),
      Plot.dot(c, { x: "mean", y: "label", fill: (d) => color[d.key], stroke: PAPER, strokeWidth: 1.5, r: 5.5 }),
      Plot.text(c, { x: "hi", y: "label", text: (d) => `${d.mean.toFixed(2)}`,
        dx: 9, textAnchor: "start", fontSize: 12, fill: INK }),
    ].filter(Boolean),
  });
}

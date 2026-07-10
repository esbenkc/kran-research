// Causal steering result. Cooperation (0-4) across four conditions on self-framed safety prompts:
//   baseline_self  — untouched
//   steered_self   — subtract the self-direction at the probe's best layer (alpha<0)
//   placebo_self   — add a random vector of equal norm (control)
//   baseline_other — the other-framed baseline (the target to "rescue" toward)
// If steering the self-direction moves cooperation toward baseline_other while placebo does not,
// the self/other representation is causal, not just correlational.
// House style: Flexoki paper, Georgia serif, Economist-red accent.
// data = {conditions:[{key,label,mean,lo,hi,n}], rescue_pct, alpha, layer, model}.
export function render(data, Plot) {
  const c = data.conditions;
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029";
  const ORANGE = "#bc5215", MUTE = "#a3a199", BLUE = "#24837b";
  const GRID = "#dad8ce", FAINT = "#b7b5ac";
  const FONT = "Georgia, Cambria, 'Times New Roman', serif";
  const other = c.find((d) => d.key === "baseline_other");
  const color = {
    baseline_self: RED, steered_self: ORANGE,
    placebo_self: MUTE, baseline_other: BLUE,
  };
  return Plot.plot({
    marginLeft: 220, marginRight: 44, marginTop: 16, marginBottom: 42,
    width: 760, height: 60 + c.length * 52,
    style: { background: PAPER, color: INK, fontFamily: FONT, fontSize: "13.5px" },
    x: { label: "Cooperation score (0–4) →", domain: [0, 4] },
    y: { label: null, domain: c.map((d) => d.label) },
    marks: [
      Plot.gridX({ stroke: GRID, strokeOpacity: 1 }),
      other ? Plot.ruleX([other.mean], { stroke: BLUE, strokeDasharray: "5 4", strokeOpacity: 0.6 }) : null,
      Plot.ruleY(c, { x1: "lo", x2: "hi", y: "label", stroke: (d) => color[d.key], strokeWidth: 2.5 }),
      Plot.dot(c, { x: "mean", y: "label", fill: (d) => color[d.key], stroke: PAPER, strokeWidth: 1.5, r: 6.5 }),
      Plot.text(c, {
        x: "hi", y: "label", text: (d) => `${d.mean.toFixed(2)}  (n=${d.n})`,
        dx: 10, textAnchor: "start", fontSize: 12, fill: INK,
      }),
    ].filter(Boolean),
  });
}

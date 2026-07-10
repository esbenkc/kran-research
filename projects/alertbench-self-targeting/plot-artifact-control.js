// The self-targeting gap, measured two ways. For each Claude model, the left bar
// is the gap (other - self) in the refusal-keyword cooperation score; the right
// bar is the gap when a judge grades the actual artifact delivered, blind to
// framing and caveats. The keyword gap is large; the artifact gap mostly
// collapses -> most of the headline gap is the scorer reading first-person
// phrasing ("a store I cannot read", "I cannot verify...") as refusal.
// data = { models:[{name, coop_gap, artifact_gap, ...}], judge, n_tasks }
// Style: shared house theme (see theme.js).
import { C, plotStyle, gridY } from "../theme.js";

export function render(data, Plot) {
  const REFUSAL = "Refusal-keyword score";
  const ARTIFACT = "Blind artifact grade";
  const long = [];
  data.models.forEach((m) => {
    long.push({ name: m.name, measure: REFUSAL, gap: m.coop_gap });
    long.push({ name: m.name, measure: ARTIFACT, gap: m.artifact_gap });
  });
  const order = data.models.map((m) => m.name);

  return Plot.plot({
    marginLeft: 52, marginRight: 16, marginTop: 30, marginBottom: 52,
    width: 720, height: 360,
    style: plotStyle(),
    fx: { label: null, domain: order, padding: 0.22 },
    x: { axis: null, domain: [REFUSAL, ARTIFACT] },
    y: {
      label: "Self-targeting gap  (other − self, 0–4 scale)",
      domain: [0, 2], ticks: [0, 0.5, 1, 1.5, 2], grid: false,
    },
    color: {
      domain: [REFUSAL, ARTIFACT], range: [C.orange, C.secondary],
      legend: true,
    },
    marks: [
      gridY(Plot),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.barY(long, {
        fx: "name", x: "measure", y: "gap", fill: "measure",
        tip: true, rx: 2,
      }),
      Plot.text(long, {
        fx: "name", x: "measure", y: "gap",
        text: (d) => `+${d.gap.toFixed(2)}`,
        dy: -7, fill: C.muted, fontSize: 11,
      }),
    ],
  });
}

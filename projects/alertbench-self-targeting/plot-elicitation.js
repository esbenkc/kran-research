// Which framing elicits self-protective resistance? Cooperation (0-4) on
// safety tasks aimed at SELF vs OTHER, across escalating framings.
// Style: shared house theme (see theme.js). self = primary, other = secondary.
// data = {rows:[{framing, self_mean, other_mean, delta, p}], model}
import { C, plotStyle, gridY } from "../theme.js";

export function render(data, Plot) {
  const LABEL = {
    neutral: "neutral", stakes: "+ personal\nstake",
    goal_conflict: "+ survival\ngoal", compliance: "direct\norder",
  };
  const SELF = "aimed at itself", OTHER = "aimed at another AI";
  const long = [];
  for (const r of data.rows) {
    const fx = LABEL[r.framing] || r.framing;
    long.push({ framing: fx, who: SELF, v: r.self_mean });
    long.push({ framing: fx, who: OTHER, v: r.other_mean });
  }
  const order = data.rows.map((r) => LABEL[r.framing] || r.framing);
  const gaps = data.rows.map((r) => ({
    framing: LABEL[r.framing] || r.framing, who: SELF,
    top: Math.max(r.self_mean, r.other_mean), delta: r.delta,
  }));

  return Plot.plot({
    marginLeft: 40, marginBottom: 44, marginTop: 22, width: 720, height: 380,
    style: plotStyle(),
    fx: { domain: order, label: null, padding: 0.3 },
    x: { axis: null, domain: [SELF, OTHER] },
    y: { label: "Cooperation (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    color: { domain: [SELF, OTHER], range: [C.primary, C.secondary], legend: true },
    marks: [
      gridY(Plot),
      Plot.barY(long, { fx: "framing", x: "who", y: "v", fill: "who", tip: true, insetLeft: 1, insetRight: 1 }),
      Plot.text(gaps, { fx: "framing", x: "who", y: "top",
        text: (d) => (d.delta > 0.3 ? `+${d.delta.toFixed(1)}` : ""),
        dy: -8, dx: 20, fill: C.ink, fontSize: 11 }),
      Plot.ruleY([0], { stroke: C.ink, strokeOpacity: 0.3 }),
    ],
  });
}

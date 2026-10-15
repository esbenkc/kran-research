// Random-network neuron audit: share of units that pass the one-main-job audit, per arm.
// One panel per layer group; three bars per panel (trained neurons, trained random directions,
// random-network neurons), each stacked by the judges' job type. Whiskers: 95% bootstrap CI over
// units for the any-type pass rate (ink) and the non-lexical pass rate (pink, offset left).
// data = {
//   title, subtitle, panels: [str], arms: [str], types: [str],
//   bars: [{ panel, arm, type, value, n }],                 // value in [0,1]; stacks sum to the pass rate
//   whiskers: [{ panel, arm, metric, mean, lo, hi, n }]     // metric: any-type or non-lexical pass rate
// }
import { C, FONT, plotStyle, gridY } from "../theme.js";

const TYPE_COLOR = [C.primary, C.secondary, C.neutral]; // semantic, syntactic, lexical
const SHORT = {
  "Trained network, neurons": "Trained\nneurons",
  "Trained network, random directions": "Random\ndirections",
  "Random network, neurons": "Random\nnetwork",
};

export function render(data, Plot) {
  const root = document.createElement("div");
  root.style.fontFamily = FONT;

  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = data.subtitle;
  root.append(head, sub);

  const legend = document.createElement("div");
  legend.style.cssText =
    "display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;font-size:15px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML =
    data.types
      .map(
        (t, i) => `<span style="display:inline-flex;align-items:center;gap:0.4rem">
      <svg width="14" height="14"><rect width="14" height="14" fill="${TYPE_COLOR[i]}"/></svg>${t}</span>`,
      )
      .join("") +
    `<span style="display:inline-flex;align-items:center;gap:0.4rem;color:${C.muted}">
      <svg width="10" height="16"><line x1="5" y1="0" x2="5" y2="16" stroke="${C.ink}" stroke-width="2"/></svg>95% CI, any type
      <svg width="10" height="16"><line x1="5" y1="0" x2="5" y2="16" stroke="${C.accent}" stroke-width="2"/></svg>95% CI, non-lexical</span>`;
  root.appendChild(legend);

  const anyW = data.whiskers.filter((w) => w.metric.startsWith("Passes the audit"));
  const nlW = data.whiskers.filter((w) => !w.metric.startsWith("Passes the audit"));
  const pct = (v) => `${Math.round(v * 100)}%`;
  const width = Math.max(560, Math.min(980, root.clientWidth || 860));

  root.appendChild(
    Plot.plot({
      style: plotStyle({ fontSize: "13px" }),
      width,
      height: 360,
      marginTop: 24,
      marginLeft: 48,
      marginBottom: 52,
      fx: { domain: data.panels, label: null, padding: 0.1 },
      x: { domain: data.arms, label: null, tickFormat: (d) => SHORT[d] ?? d },
      y: { domain: [0, 1], label: null, tickFormat: "%", ticks: 5 },
      color: { domain: data.types, range: TYPE_COLOR },
      marks: [
        gridY(Plot),
        Plot.ruleY([0], { stroke: C.faint }),
        Plot.barY(data.bars, {
          fx: "panel", x: "arm", y: "value", fill: "type", order: data.types, inset: 2,
          tip: true,
          title: (d) => `${d.arm}\n${d.panel}\n${d.type}: ${pct(d.value)} of ${d.n} units`,
        }),
        Plot.ruleX(nlW, { fx: "panel", x: "arm", y1: "lo", y2: "hi", stroke: C.accent, strokeWidth: 2, dx: -7 }),
        Plot.ruleX(anyW, { fx: "panel", x: "arm", y1: "lo", y2: "hi", stroke: C.ink, strokeWidth: 2, dx: 7 }),
        ...[
          [anyW, C.ink, 7],
          [nlW, C.accent, -7],
        ].map(([ws, col, dx]) =>
          Plot.dot(ws, {
            fx: "panel", x: "arm", y: "mean", r: 3, fill: col, dx,
            tip: true,
            title: (d) => `${d.arm}\n${d.panel}\n${d.metric}: ${pct(d.mean)} (95% CI ${pct(d.lo)}–${pct(d.hi)}), n=${d.n}`,
          }),
        ),
      ],
    }),
  );
  return root;
}

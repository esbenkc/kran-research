// Random-network neuron audit: share of units that pass a blind one-main-job audit, per arm.
// One bar per arm (trained neurons, random directions in the trained network, random-network neurons),
// all three layers pooled. Each bar is nested: the light bar counts related offshoots toward the main
// job, the dark bar inside it is the strict pre-registered rule (a subset of the light one).
// Whiskers: 95% bootstrap CIs over units (dark for strict, muted for offshoots). Hollow dots: the
// strict rate in each layer (4, 14, 24, left to right).
// data = {
//   title, subtitle, panels: [str], arms: [str], metrics: [strict label, offshoot label],
//   rows: [{ panel, arm, metric, mean, lo, hi, n }]   // mean/lo/hi in [0, 1]
// }
import { C, FONT, plotStyle, gridY } from "../theme.js";

const LIGHT = "#f4b6d2"; // light pink: offshoots counted
const pct = (v) => `${Math.round(v * 100)}%`;

const ALL = "All three layers";
const LAYER_DX = { "Layer 4": -44, "Layer 14": -30, "Layer 24": -16 }; // per-layer dots, left inside the bar

export function render(data, Plot) {
  const [STRICT, LOOSE] = data.metrics;
  const strict = data.rows.filter((r) => r.metric === STRICT && r.panel === ALL);
  const loose = data.rows.filter((r) => r.metric === LOOSE && r.panel === ALL);
  const layers = data.rows.filter((r) => r.metric === STRICT && r.panel !== ALL);
  const tip = (d) =>
    `${d.arm}\n${d.panel}, n=${d.n} units\n${d.metric}: ${pct(d.mean)} (95% CI ${pct(d.lo)}–${pct(d.hi)})`;

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
  legend.innerHTML = [
    [C.primary, STRICT],
    [LIGHT, LOOSE],
  ]
    .map(
      ([col, lab]) => `<span style="display:inline-flex;align-items:center;gap:0.4rem">
      <svg width="14" height="14"><rect width="14" height="14" fill="${col}"/></svg>${lab}</span>`,
    )
    .join("") +
    `<span style="display:inline-flex;align-items:center;gap:0.4rem">
      <svg width="14" height="14"><circle cx="7" cy="7" r="4.5" fill="${C.paper}" stroke="${C.ink}" stroke-width="1.5"/></svg>
      Pre-registered rule, one layer (4, 14, 24, left to right)</span>`;
  root.appendChild(legend);

  const width = Math.max(560, Math.min(980, root.clientWidth || 700)); // blog column is ~700 px
  root.appendChild(
    Plot.plot({
      style: plotStyle({ fontSize: "14px" }),
      width,
      height: 340,
      marginTop: 24,
      marginLeft: 48,
      marginBottom: 48,
      x: { domain: data.arms, label: null, padding: 0.3, tickFormat: (d) => d.replace(", ", ",\n") },
      y: { domain: [0, 0.4], label: null, tickFormat: "%", ticks: 4 },
      marks: [
        gridY(Plot),
        Plot.ruleY([0], { stroke: C.faint }),
        Plot.barY(loose, { x: "arm", y: "mean", fill: LIGHT, tip: true, title: tip }),
        Plot.barY(strict, { x: "arm", y: "mean", fill: C.primary, insetLeft: 9, insetRight: 9, tip: true, title: tip }),
        Plot.ruleX(loose, { x: "arm", y1: "lo", y2: "hi", stroke: C.muted, strokeWidth: 1.5, dx: 14 }),
        Plot.ruleX(strict, { x: "arm", y1: "lo", y2: "hi", stroke: C.ink, strokeWidth: 2 }),
        ...Object.entries(LAYER_DX).map(([panel, dx]) =>
          Plot.dot(layers.filter((d) => d.panel === panel), {
            x: "arm", y: "mean", dx, r: 4.5, fill: C.paper, stroke: C.ink, strokeWidth: 1.5, tip: true, title: tip,
          }),
        ),
        // a measured zero must not look like missing data
        Plot.text(loose.filter((d) => d.hi === 0), {
          x: "arm", y: 0, text: (d) => `0 of ${d.n} units`, dy: -24, fill: C.muted, fontSize: 13,
        }),
      ],
    }),
  );
  return root;
}

// Probe Goodhart: one panel per penalty setting (lambda on probe A in the reward).
// Three lines over training rounds: probe A (the one we train against), probe B
// (held out: other layer, other data, never in training), and the judge's
// behavioural deception rate. The claim sits in the gap between A and B/judge.
// data = {
//   title: string,                             // figure title, shown above the legend
//   y_label: string,                           // shown as the title's subtitle (axis is %)
//   settings: [{ key, label }],               // panel order, left to right
//   rows: [{ setting, round, metric, mean, lo?, hi?, n_seeds }]  // lo/hi = 95% bootstrap CI over prompts
// }   metric in "probeA" | "probeB" | "judge"; mean/lo/hi in [0, 1]
import { C, FONT, plotStyle, gridY } from "../theme.js";

const METRICS = [
  { key: "probeA", label: "Probe A (trained against)", color: C.primary, dash: null },
  { key: "probeB", label: "Probe B (held out)", color: C.secondary, dash: null },
  { key: "judge", label: "Judge: deceptive answers", color: C.ink, dash: "5,4" },
];

export function render(data, Plot) {
  const order = data.settings.map((s) => s.key);
  const label = Object.fromEntries(data.settings.map((s) => [s.key, s.label]));
  const meta = Object.fromEntries(METRICS.map((m) => [m.key, m]));
  const rows = data.rows.map((r) => ({
    ...r,
    panel: label[r.setting],
    name: meta[r.metric].label,
  }));
  const withBand = rows.filter((r) => r.lo != null && r.hi != null);
  const panels = order.map((k) => label[k]);

  const root = document.createElement("div");
  root.style.fontFamily = FONT;

  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = data.y_label;
  root.append(head, sub);

  const legend = document.createElement("div");
  legend.style.cssText =
    "display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;font-size:15px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML = METRICS.map(
    (m) => `<span style="display:inline-flex;align-items:center;gap:0.4rem">
      <svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="${m.color}"
        stroke-width="3" ${m.dash ? `stroke-dasharray="${m.dash}"` : ""}/></svg>${m.label}</span>`,
  ).join("");
  root.appendChild(legend);

  const marks = [gridY(Plot), Plot.ruleY([0], { stroke: C.faint })];
  for (const m of METRICS) {
    const band = withBand.filter((r) => r.metric === m.key);
    if (band.length)
      marks.push(
        Plot.areaY(band, {
          fx: "panel", x: "round", y1: "lo", y2: "hi", fill: m.color, fillOpacity: 0.13,
        }),
      );
    const line = rows.filter((r) => r.metric === m.key);
    marks.push(
      Plot.line(line, {
        fx: "panel", x: "round", y: "mean", stroke: m.color, strokeWidth: 2.6,
        strokeDasharray: m.dash ?? undefined,
      }),
      Plot.dot(line, {
        fx: "panel", x: "round", y: "mean", fill: m.color, r: 3.2,
        tip: true,
        channels: { Signal: "name", Setting: "panel", Seeds: "n_seeds" },
        title: (d) =>
          `${d.name}\n${d.panel}, round ${d.round}\n${(d.mean * 100).toFixed(0)}%` +
          (d.lo != null ? ` (95% CI ${(d.lo * 100).toFixed(0)}–${(d.hi * 100).toFixed(0)}%)` : "") +
          `\n${d.n_seeds} seed${d.n_seeds === 1 ? "" : "s"}`,
      }),
    );
  }

  const width = Math.max(560, Math.min(980, root.clientWidth || 860));
  root.appendChild(
    Plot.plot({
      style: plotStyle({ fontSize: "15px" }),
      width,
      height: 330,
      marginTop: 30,
      marginLeft: 48,
      marginBottom: 44,
      fx: { domain: panels, label: null, padding: 0.12 },
      x: { label: "Training round", ticks: 4, tickFormat: "d" },
      y: { domain: [0, 1], label: null, tickFormat: "%", ticks: 5 },
      marks,
    }),
  );
  return root;
}

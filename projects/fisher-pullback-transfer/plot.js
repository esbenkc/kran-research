// Fisher pull-back transfer: measured AUROC drop of each held-out probe vs the drop predicted from one
// formula, after a behaviour-preserving fine-tune hid one concept from one probe.
// Three panels = three metrics in the same formula (plain, whitened, Fisher pull-back).
// Filled dots: the pre-registered set (5 targets x 49 probes of OTHER concepts, mean of seeds).
// Hollow dots: the secondary same-concept stack (added before the attacks; never pooled with the first set).
// Dashed line: prediction = measurement.
// data = {
//   title, sub, scale: "linear" | "symlog", symlog_constant,
//   panels: [{ metric, label, r2_primary, r2_secondary }],
//   stats: { n_primary, sd_primary, n02_primary, n_secondary, sd_secondary, n02_secondary },
//   rows: [{ set: "primary"|"secondary", metric, target, probe, x, y, n_seeds }]
// }   x = predicted AUROC drop, y = measured AUROC drop (AUROC clean minus AUROC after the attack)
import { C, FONT, plotStyle } from "../theme.js";

const fmtR = (v) => (v === null || Number.isNaN(v) ? "n/a" : (v < -9.95 ? "< −10" : v.toFixed(2).replace("-", "−")));

export function render(data, Plot) {
  const SET = {
    primary: { label: "other concepts (pre-registered test)", color: C.primary, filled: true },
    secondary: { label: "same concept, other probes (secondary)", color: C.secondary, filled: false },
  };
  const root = document.createElement("div");
  root.style.fontFamily = FONT;

  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = data.sub;

  const legend = document.createElement("div");
  legend.style.cssText =
    "display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;font-size:15px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML =
    `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="${SET.primary.color}"/></svg>filled dot: ${SET.primary.label}, n = ${data.stats.n_primary}</span>` +
    `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="none" stroke="${SET.secondary.color}" stroke-width="1.8"/></svg>hollow dot: ${SET.secondary.label}, n = ${data.stats.n_secondary}</span>` +
    `<span style="display:inline-flex;align-items:center;gap:0.4rem;color:${C.muted}"><svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" stroke="${C.faint}" stroke-width="2" stroke-dasharray="4,3"/></svg>prediction = measurement</span>`;
  root.append(head, sub, legend);

  const grid = document.createElement("div");
  grid.style.cssText = "display:flex;flex-wrap:wrap;gap:0.5rem 12px";
  root.appendChild(grid);

  const vals = data.rows.flatMap((r) => [r.x, r.y]);
  const lo = Math.min(-0.05, Math.floor(Math.min(...vals) * 20) / 20);
  const hi = Math.max(0.05, Math.ceil(Math.max(...vals) * 20) / 20);
  const scale = data.scale === "symlog"
    ? { type: "symlog", constant: data.symlog_constant, domain: [lo, hi] }
    : { domain: [lo, hi] };
  const ticks = data.ticks || null;

  // Three 178 px panels fit the 558 px figure column; the root's width attribute tells the /lab
  // wall (assets/js/lab-wall.js) the natural width to scale from.
  const width = 178;
  root.setAttribute("width", String(3 * width + 24));

  for (const panel of data.panels) {
    const box = document.createElement("div");
    box.style.cssText = `width:${width}px;flex:0 0 auto`;
    const ph = document.createElement("div");
    ph.style.cssText = "font-size:15px;font-weight:600;margin:0.2rem 0 0.1rem;color:" + C.ink;
    ph.textContent = panel.label;
    const stats = document.createElement("div");
    stats.style.cssText = "font-size:13px;line-height:1.35;margin:0 0 0.2rem;color:" + C.muted;
    stats.innerHTML =
      `R² filled: <b style="color:${C.ink}">${fmtR(panel.r2_primary)}</b><br>` +
      `R² hollow: <b style="color:${C.ink}">${fmtR(panel.r2_secondary)}</b>`;
    box.append(ph, stats);

    const rows = data.rows.filter((r) => r.metric === panel.metric);
    const tip = (d) =>
      `${d.probe}\nattack hid: ${d.target}\n` +
      `predicted drop: ${d.x.toFixed(3)}\nmeasured drop: ${d.y.toFixed(3)} (mean of ${d.n_seeds} seeds)`;
    box.appendChild(Plot.plot({
      style: plotStyle({ fontSize: "12px" }), width, height: width + 20,
      marginTop: 8, marginLeft: 44, marginBottom: 40, marginRight: 8,
      x: { ...scale, ticks: ticks || 4, label: "Predicted drop", labelAnchor: "center", labelArrow: "none" },
      y: { ...scale, ticks: ticks || 4, label: "Measured drop", labelAnchor: "center", labelArrow: "none" },
      marks: [
        Plot.gridY(ticks || 4, { stroke: C.grid, strokeOpacity: 1 }),
        Plot.ruleX([0], { stroke: C.faint, strokeOpacity: 0.6 }),
        Plot.line([[lo, lo], [hi, hi]], { stroke: C.faint, strokeWidth: 1.5, strokeDasharray: "5,4" }),
        Plot.dot(rows.filter((r) => r.set === "primary"), {
          x: "x", y: "y", r: 2.8, fill: SET.primary.color, fillOpacity: 0.8, stroke: C.paper, strokeWidth: 0.5,
          tip: true, title: tip,
        }),
        Plot.dot(rows.filter((r) => r.set === "secondary"), {
          x: "x", y: "y", r: 3.4, fill: "none", stroke: SET.secondary.color, strokeWidth: 1.4,
          tip: true, title: tip,
        }),
      ],
    }));
    grid.appendChild(box);
  }
  return root;
}

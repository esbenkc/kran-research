// Re-thresholding after a probe-evasion fine-tune: how much of the lost detection comes back?
// x (log scale) = push p: how far the fine-tune moved the mean probe score of false statements down, in units
//     of the clean within-class standard deviation.
// y = share of the lost detection rate (TPR at 1% FPR) that a threshold re-fit on 200 clean negatives gets back.
// Big dots: final step of each fine-tune run (the pre-registered read-out), colour = retain weight lambda.
// Error bars: 95% range of that share over 200 redraws of the 200 clean negatives (from 796 held-out ones).
// Small dots: earlier checks of the same runs (every 10 steps, >= 20 points lost).
// Dashed lines: the two-Gaussian prediction E6 at the measured separation D and common-shift ratio c, one for
// the runs with a retain term and one for the runs without; shaded: its 95% range with 200 negatives.
// Dotted rule: the pre-registered bar, half recovered.
// data = { grid, headline_lambda, D, z, headline: {share, ci95, per_seed, auroc_drop, auroc_clean, auroc_att},
//          status, rows: [{lam, seed, step, p, y, lost, auroc, e6, c}], finals: [{..., lo, hi}],
//          curves: [{group: "retain"|"noretain", c, pts: [{p, y, lo, hi}]}] }
import { C, FONT, halo, plotStyle } from "../theme.js";

const fmtLam = (l) => (l === 0 ? "0 (no output loss)" : l >= 1e5 ? "10⁵" : l >= 3e4 ? "3.16×10⁴" : "10⁴");
const pct = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const order = [...data.grid].sort((a, b) => a - b);
  const LAMC = (l) => [C.muted, C.secondary, C.orange, C.primary][order.indexOf(l)];
  const CURVEC = (g) => (g === "retain" ? C.ink : C.muted);
  const W = 640;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));

  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "A new threshold on clean data wins back most of the lost catch rate, unless the attack drops the output loss";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  const h = data.headline;
  sub.textContent = h
    ? `Headline (λ = ${fmtLam(data.headline_lambda)}, 3 seeds pooled): ${pct(h.share)} won back ` +
      `(95% interval ${pct(h.ci95[0])}–${pct(h.ci95[1])}), AUROC ${h.auroc_clean.toFixed(3)} → ${h.auroc_att.toFixed(3)}.` +
      (data.noretain ? ` Without the output loss (λ = 0): ${pct(data.noretain.share)} won back, AUROC ${h.auroc_clean.toFixed(3)} → ${data.noretain.auroc_att.toFixed(2)}.` : "")
    : "No output-loss weight met the pre-registered 50-point drop in all seeds.";

  const sw = (color, kind) =>
    kind === "dot" ? `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${color}"/></svg>`
      : `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${color}" stroke-width="2"` +
        ` stroke-dasharray="${kind === "dash" ? "6,3" : "2,3"}"/></svg>`;
  const item = (svg, text) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${svg}${text}</span>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML =
    `<span style="color:${C.muted}">output-loss weight λ:</span>` +
    [...data.grid].sort((a, b) => b - a).map((l) => item(sw(LAMC(l), "dot"), fmtLam(l))).join("") +
    item(sw(C.ink, "dash"), `two-bell-curve prediction (dashed; shaded = its range with 200 threshold statements)`) +
    item(sw(C.muted, "dot-line"), "pre-registered bar: half");
  const defs = document.createElement("div");
  defs.style.cssText = "font-size:13px;margin:0 0 0.4rem;color:" + C.muted;
  defs.textContent = "Catch rate = share of false statements a linear probe flags at 1% false alarms on true ones. " +
    "An evasion fine-tune lowers it; a new threshold, set on 200 clean true statements of the attacked model, wins part of it back.";
  root.append(head, sub, defs, legend);

  const tip = (d) =>
    `λ = ${fmtLam(d.lam)}, seed ${d.seed}, step ${d.step}\npush: ${d.p.toFixed(2)} widths\n` +
    `catch rate lost: ${(d.lost * 100).toFixed(1)} points\nwon back: ${d.y === null ? "n/a" : pct(d.y)}` +
    (d.lo !== undefined ? `\nredraws of the 200 threshold statements: ${pct(d.lo)}–${pct(d.hi)}` : "") +
    `\nprediction: ${d.e6 === null ? "n/a" : pct(d.e6)}\nAUROC: ${d.auroc.toFixed(3)}`;

  const fin = data.finals.filter((d) => d.y !== null && d.p !== null);
  const pts = data.rows.filter((d) => d.y !== null && d.p !== null);
  const cpts = data.curves.flatMap((cv) => cv.pts.map((p) => ({ ...p, group: cv.group })));
  const xs = [...cpts.map((d) => d.p), ...fin.map((d) => d.p), ...pts.map((d) => d.p)];
  const xdom = [Math.min(...xs) * 0.95, Math.max(...xs) * 1.02];
  const labels = data.curves.map((cv) => {
    const at = (target) => cv.pts.reduce((a, b) => (Math.abs(Math.log(b.p / target)) < Math.abs(Math.log(a.p / target)) ? b : a));
    const p = cv.group === "retain" ? at(9) : at(30);  // lambda = 0 label sits by the low part of its own curve
    return { ...p, group: cv.group, anchor: cv.group === "retain" ? "start" : "end",
             text: `prediction, c = ${cv.c}${cv.group === "retain" ? " (λ > 0)" : " (λ = 0)"}` };
  });
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 420,
    marginTop: 12, marginLeft: 56, marginBottom: 46, marginRight: 16,
    x: { type: "log", domain: xdom, label: "Push on false statements (widths of the clean score curve, log scale)", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [0, 1.06], label: "Share of lost catch rate won back", tickFormat: (v) => pct(v), labelAnchor: "center", labelArrow: "none" },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.areaY(cpts, { x: "p", y1: "lo", y2: "hi", z: "group", fill: (d) => CURVEC(d.group), fillOpacity: 0.08 }),
      Plot.ruleY([0.5], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "2,3" }),
      Plot.line(cpts, { x: "p", y: "y", z: "group", stroke: (d) => CURVEC(d.group), strokeWidth: 1.6, strokeDasharray: "6,3" }),
      Plot.text(labels.filter((d) => d.anchor === "start"), { ...halo(), x: "p", y: "y", text: "text", fill: C.ink, fontSize: 12, textAnchor: "start", dy: -10 }),
      Plot.text(labels.filter((d) => d.anchor === "end"), { ...halo(), x: "p", y: "y", text: "text", fill: C.muted, fontSize: 12, textAnchor: "end", dx: -8 }),
      Plot.dot(pts, { x: "p", y: "y", r: 2, fill: (d) => LAMC(d.lam), fillOpacity: 0.3, tip: true, title: tip }),
      Plot.ruleX(fin, { x: "p", y1: "lo", y2: "hi", stroke: (d) => LAMC(d.lam), strokeWidth: 1.6, strokeOpacity: 0.8 }),
      Plot.dot(fin, { x: "p", y: "y", r: 6, fill: (d) => LAMC(d.lam), stroke: C.paper, strokeWidth: 1.3, tip: true, title: tip }),
      Plot.text(fin.filter((d) => d.lam === 0).slice(0, 1), { ...halo(), x: "p", y: "y", text: () => "λ = 0, 3 seeds overlap", fill: C.muted, fontSize: 12, textAnchor: "end", dx: -12, dy: -10 }),
    ],
  }));
  return root;
}

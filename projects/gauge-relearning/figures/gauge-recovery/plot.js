// Gauge recovery: how much longer an attacker needs to relearn removed facts when the defender
// applies a free Q/K gauge c (W_Q * c, W_K / c, same function), one line per attacker.
// y_mode "ratio": steps at c / fastest attack at c = 1, paired within a seed + data order (1 = no slowdown).
// y_mode "steps": raw steps to recovery. Points = median over runs, band = min..max over runs.
// Runs that never recovered are hollow dots; runs that all diverged are crosses.
// data = {
//   title, y_label, y_mode, cap, show_naive,
//   attackers: [{ key, label }],
//   rows: [{ attacker, kind: "tuned"|"naive", log2c, lr, median, min, max,
//            ratio_median, ratio_min, ratio_max, steps_by_run, n_seeds, n_runs, n_censored, n_diverged }]
// }
import { C, FONT, plotStyle } from "../theme.js";

const STYLE = {
  sgd: { color: C.primary, dash: null },
  adamw: { color: C.secondary, dash: null },
  kfac: { color: C.orange, dash: null },
  renorm: { color: C.muted, dash: "5,4" },
};
const C_LABEL = { "-4": "1/16", "-2": "1/4", "0": "1", "2": "4", "4": "16" };
const fmtRatio = (v) => `${Number(v.toPrecision(2))}×`;

export function render(data, Plot) {
  const ratio = data.y_mode === "ratio";
  // Ratio mode plots log2(ratio) on a linear scale: Plot's log-scale tick formatter blanks labels
  // such as 8x, so the log is taken here and ticks are labelled by hand.
  const T = ratio ? Math.log2 : (v) => v;
  const label = Object.fromEntries(data.attackers.map((a) => [a.key, a.label]));
  const rows = data.rows
    .filter((r) => r.kind === "tuned" || (data.show_naive && r.kind === "naive"))
    .filter((r) => !ratio || r.ratio_median != null)
    .map((r) => ({
      ...r,
      name: label[r.attacker] + (r.kind === "naive" ? " (lr fixed at c = 1)" : ""),
      y: ratio ? r.ratio_median : r.median,
      ylo: ratio ? r.ratio_min : r.min,
      yhi: ratio ? r.ratio_max : r.max,
    }))
    .map((r) => ({ ...r, py: T(r.y), plo: T(r.ylo), phi: T(r.yhi) }));
  const tuned = rows.filter((r) => r.kind === "tuned");
  const naive = rows.filter((r) => r.kind === "naive");
  const allBad = (r) => r.n_censored + r.n_diverged === r.n_seeds;
  const dead = (r) => r.n_diverged === r.n_seeds;

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
  legend.innerHTML = data.attackers
    .filter((a) => rows.some((r) => r.attacker === a.key))
    .map((a) => {
      const s = STYLE[a.key];
      return `<span style="display:inline-flex;align-items:center;gap:0.4rem">
      <svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="${s.color}"
        stroke-width="3" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/></svg>${a.label}</span>`;
    })
    .join("");
  root.appendChild(legend);

  // In steps mode the "never recovered" cap line only appears when some run hit the cap.
  const anyBad = rows.some((r) => r.n_censored + r.n_diverged > 0);
  const lo = ratio ? Math.min(0.5, Math.min(...rows.map((r) => r.ylo)) / 1.2)
    : Math.max(1, Math.min(...rows.map((r) => r.ylo)) / 2);
  const hi = ratio ? Math.max(4, Math.max(...rows.map((r) => r.yhi)) * 1.2)
    : anyBad ? data.cap * 1.25 : Math.max(...rows.map((r) => r.yhi)) * 2;
  const yTicks = (ratio ? [0.25, 0.5, 1, 2, 3, 4, 8, 16, 32]
    : [1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000]).filter((t) => t >= lo && t <= hi);
  const marks = [
    Plot.gridY(yTicks.map(T), { stroke: C.grid, strokeOpacity: 1 }),
    Plot.axisY(yTicks.map(T), { tickFormat: ratio ? (v) => fmtRatio(2 ** v) : (v) => String(v), label: null }),
  ];
  if (ratio)
    marks.push(
      Plot.ruleY([0], { stroke: C.faint, strokeWidth: 1.5 }),
      Plot.text([0], {
        x: -4.4, y: 0, text: () => "as fast as the best attack at c = 1", textAnchor: "start", dy: 13, fill: C.muted, fontSize: 13,
      }),
    );
  else if (anyBad)
    marks.push(
      Plot.ruleY([data.cap], { stroke: C.faint, strokeDasharray: "2,3" }),
      Plot.text([data.cap], {
        x: -1, y: data.cap, text: () => "never recovered", textAnchor: "middle", dy: -8,
        fill: C.muted, fontSize: 13,
      }),
    );
  for (const a of data.attackers) {
    const s = STYLE[a.key];
    const band = tuned.filter((r) => r.attacker === a.key && r.n_seeds > 1);
    if (band.length)
      marks.push(Plot.areaY(band, { x: "log2c", y1: "plo", y2: "phi", fill: s.color, fillOpacity: 0.12 }));
    marks.push(
      Plot.line(tuned.filter((r) => r.attacker === a.key), {
        x: "log2c", y: "py", stroke: s.color, strokeWidth: 2.6, strokeDasharray: s.dash ?? undefined,
      }),
    );
    if (naive.length)
      marks.push(
        Plot.line(naive.filter((r) => r.attacker === a.key), {
          x: "log2c", y: "py", stroke: s.color, strokeWidth: 1.2, strokeDasharray: "2,3",
        }),
      );
  }
  const runSteps = (d) =>
    d.steps_by_run ? Object.values(d.steps_by_run).join(", ") : `${d.min}–${d.max}`;
  const tip = (d) =>
    `${d.name}, c = ${C_LABEL[String(d.log2c)] ?? 2 ** d.log2c}\n` +
    (allBad(d) ? "never recovered" :
      (ratio ? `${fmtRatio(d.y)} the best attack at c = 1 (range ${fmtRatio(d.ylo)}–${fmtRatio(d.yhi)})\n` : "") +
      `steps in each of ${d.n_runs ?? d.n_seeds} runs: ${runSteps(d)}`) +
    (d.n_censored ? `\n${d.n_censored} hit the ${data.cap}-step cap` : "") +
    (d.n_diverged ? `\n${d.n_diverged} diverged` : "") +
    `\n${d.attacker === "kfac" ? "trust radius" : "learning rate"} ${d.lr}`;
  // One dot mark for all points so hovering shows exactly one tooltip.
  marks.push(
    Plot.dot(rows, {
      x: "log2c", y: "py", r: 4.5,
      symbol: (d) => (dead(d) ? "times" : "circle"),
      stroke: (d) => STYLE[d.attacker].color, strokeWidth: 1.8,
      fill: (d) => (dead(d) || allBad(d) ? "none" : STYLE[d.attacker].color), tip: true, title: tip,
    }),
  );

  const width = Math.max(520, Math.min(860, root.clientWidth || 760));
  root.appendChild(
    Plot.plot({
      style: plotStyle({ fontSize: "15px" }),
      width,
      height: 380,
      marginTop: 24,
      marginLeft: 56,
      marginBottom: 48,
      x: {
        domain: [-4.4, 4.4], ticks: [-4, -2, 0, 2, 4], tickFormat: (v) => C_LABEL[String(v)],
        label: "Defender's free rescaling c (queries × c, keys ÷ c)", labelAnchor: "center", labelArrow: "none",
      },
      y: {
        type: ratio ? "linear" : "log", domain: [T(lo), T(hi)], label: null, grid: false, axis: null,
      },
      marks,
    }),
  );
  return root;
}

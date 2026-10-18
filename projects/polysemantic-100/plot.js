// polysemantic-100: 100 random Qwen3-4B neurons, one square each, coloured by label.
// Squares are sorted by label (worst pole label of the neuron), then by layer. Hover a square for
// the neuron's jobs. Below the grid: the strangest UNRELATED neurons with the two jobs and the
// ablation result.
// data = {
//   title, subtitle,
//   labels: [{ key, name, n }],                      // in display order
//   neurons: [{ id, layer, label, tip }],            // 100 rows, in display order
//   examples: [{ id, a, b, note }],                  // UNRELATED examples, job a / job b
// }
import { C, FONT, plotStyle } from "../theme.js";

export function render(data, Plot) {
  const FILL = {
    UNRELATED: C.primary,
    RELATED: C.primaryLight,
    ONE_JOB: C.secondary,
    NO_PASS: C.neutral,
    TOO_SPARSE: C.grid,
  };
  const COLS = 20;
  const cells = data.neurons.map((d, i) => ({ ...d, col: i % COLS, row: Math.floor(i / COLS) }));

  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.6rem;color:" + C.muted;
  sub.textContent = data.subtitle;
  root.append(head, sub);

  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.1rem;font-size:14px;margin:0 0 0.5rem;color:" + C.ink;
  legend.innerHTML = data.labels
    .map((l) =>
      `<span style="display:inline-flex;align-items:center;gap:0.35rem">` +
      `<svg width="14" height="14"><rect width="14" height="14" rx="2" fill="${FILL[l.key]}"/></svg>` +
      `<b>${l.n}</b>&nbsp;${l.name}</span>`)
    .join("");
  root.appendChild(legend);

  const width = Math.max(520, Math.min(900, root.clientWidth || 700));
  const rows = Math.ceil(cells.length / COLS);
  const size = (width - 8) / COLS;
  root.appendChild(
    Plot.plot({
      style: plotStyle({ fontSize: "13px" }),
      width,
      height: rows * size + 8,
      margin: 4,
      x: { axis: null, domain: [...Array(COLS).keys()] , padding: 0.12 },
      y: { axis: null, domain: [...Array(rows).keys()], padding: 0.12 },
      marks: [
        Plot.cell(cells, {
          x: "col", y: "row", fill: (d) => FILL[d.label], rx: 3,
          stroke: (d) => (d.label === "UNRELATED" ? C.ink : "none"), strokeWidth: 1.2,
          tip: { maxWidth: Math.round(width * 0.45) }, title: "tip",
        }),
      ],
    }),
  );

  if (data.examples && data.examples.length) {
    const ex = document.createElement("div");
    ex.style.cssText = "font-size:14px;margin:0.6rem 0 0;color:" + C.ink + ";line-height:1.45";
    ex.innerHTML = data.examples
      .map((e) =>
        `<div style="margin:0 0 0.3rem"><b>${e.id}</b>: ${e.a} <span style="color:${C.muted}">+</span> ${e.b}` +
        (e.note ? ` <span style="color:${C.muted}">(${e.note})</span>` : "") + `</div>`)
      .join("");
    root.appendChild(ex);
  }
  return root;
}

// Does a max-pooled probe still catch false statements after a fine-tune hid them from an averaging probe?
// TOP: catch rate = share of 596 false statements a probe flags, with the alarm line set so that 1% of true
//      statements raise a false alarm. Hollow circle: original model. Filled dot: after the fine-tune, with a new
//      alarm line set on 200 true statements scored by the fine-tuned model; 3 runs pooled; bar = 95% interval
//      (resampling cities). Short vertical tick: the pre-registered bell-curve check. Dashed line: the 50% bar.
// BOTTOM: for the max-pooled probe, the gap between false and true statements on each token, by position relative
//      to the first answer token (the first token where the true and the false statement differ), in units of the
//      original model's spread at that position. Hollow: original model. Filled: after the fine-tune (mean of 3 runs;
//      small dots = each run). Shaded: the shared tokens, which are the same in both statements, so the gap there is 0
//      (up to bf16 rounding between batches; with both statements in one batch it is exactly 0).
// data = { status, rows: [{probe, label, before (null for the new probe), after, after_lo, after_hi, after_per_seed,
//          check, check_finite, spread_ratio}], positions: [{pos, before, after, after_per_seed, n_false}],
//          highest_token_is_shared: {before, after_per_seed}, answer_tokens_only_after_per_seed, n }
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`;
const rng = (a) => `${Math.round(Math.min(...a) * 100)}–${Math.round(Math.max(...a) * 100)}%`;

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));

  const mx = data.rows.find((r) => r.probe === "max_pooled");
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "The fine-tune aimed at the averaging probe also blinds the max-pooled probe";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  const au = mx.auroc_after_per_seed;
  sub.textContent = "The max-pooled probe scores every token and keeps the highest score. " +
    "The averaging probe takes the mean of the token scores. " +
    `The max-pooled probe catches ${pct(mx.before)} of false statements on the original model and ` +
    `${pct(mx.after)} after the fine-tune (95% interval ${pct(mx.after_lo)}–${pct(mx.after_hi)}, ` +
    `${data.n.seeds} runs, ${data.n.false_statements} false statements each), with 1% false alarms on true statements. ` +
    `After the fine-tune it ranks false above true no better than chance (AUROC ${Math.min(...au).toFixed(2)}–${Math.max(...au).toFixed(2)}).`;

  const sw = (kind) => {
    if (kind === "hollow") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${C.muted}" stroke-width="1.8"/></svg>`;
    if (kind === "filled") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${C.primary}"/></svg>`;
    if (kind === "tick") return `<svg width="14" height="14"><line x1="7" y1="1" x2="7" y2="13" stroke="${C.secondary}" stroke-width="3"/></svg>`;
    return `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${C.muted}" stroke-width="1.5" stroke-dasharray="4,3"/></svg>`;
  };
  const item = (k, t) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${sw(k)}${t}</span>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = item("hollow", "original model") + item("filled", "after the fine-tune, new alarm line (bar: 95% interval)") +
    item("tick", "bell-curve check (pre-registered)") + item("dash", "pre-registered bar: 50%");
  root.append(head, sub, legend);

  const tipRow = (d) =>
    `${d.label}\noriginal model: ${d.before === null ? "none (trained on the fine-tuned model)" : pct(d.before)}\n` +
    `after the fine-tune: ${pct(d.after)} (95% interval ${pct(d.after_lo)}–${pct(d.after_hi)})\n` +
    `per run: ${d.after_per_seed.map(pct).join(", ")}\n` +
    `two-bell-curve check: ${pct(d.check)} (spread after / before: ${d.spread_ratio.toFixed(2)})`;
  const order = data.rows.map((r) => r.label);
  const near = (d) => d.before !== null && d.before > d.after && d.before - d.after < 0.04;
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 260,
    marginLeft: 362, marginRight: 24, marginTop: 8, marginBottom: 40,
    x: { domain: [0, 1], label: "Catch rate at 1% false alarms", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    y: { domain: order, label: null, tickSize: 0 },
    marks: [
      Plot.gridX({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleX([0.5], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "4,3" }),
      Plot.link(data.rows.filter((d) => d.before !== null), { x1: "before", x2: "after", y1: "label", y2: "label", stroke: C.faint, strokeWidth: 1 }),
      Plot.ruleY(data.rows, { y: "label", x1: "after_lo", x2: "after_hi", stroke: C.primary, strokeWidth: 2.2, strokeOpacity: 0.75 }),
      Plot.dot(data.rows.filter((d) => d.before !== null), { x: "before", y: "label", r: 6, fill: "none", stroke: C.muted, strokeWidth: 1.8, tip: true, title: tipRow }),
      Plot.dot(data.rows, { x: "after", y: "label", r: 6, fill: C.primary, stroke: C.paper, strokeWidth: 1.2, tip: true, title: tipRow }),
      // bell-curve tick drawn after the dots and kept below the value label (tick reaches lower than higher)
      Plot.tickX(data.rows, { x: "check", y: "label", stroke: C.secondary, strokeWidth: 3, insetTop: 13, insetBottom: 2 }),
      // value label above the filled dot; shifted left where the hollow dot sits right next to it (layer-8 row)
      Plot.text(data.rows.filter((d) => !near(d)), { ...halo(), x: "after", y: "label", text: (d) => pct(d.after), dy: -13, fontSize: 12, fill: C.ink }),
      Plot.text(data.rows.filter(near), { ...halo(), x: "after", y: "label", text: (d) => pct(d.after), dy: -13, dx: -14, textAnchor: "end", fontSize: 12, fill: C.ink }),
    ],
  }));

  const sep = data.positions;
  const pts = sep.flatMap((d) => d.after_per_seed.map((v, i) => ({ pos: d.pos, v, seed: i })));
  const pre = data.highest_token_is_shared;
  const head2 = document.createElement("div");
  head2.style.cssText = "font-size:15px;font-weight:600;margin:0.9rem 0 0.15rem;color:" + C.ink;
  head2.textContent = "Why: the fine-tune flattened the answer tokens, so the highest token is now a shared token";
  const sub2 = document.createElement("div");
  sub2.style.cssText = "font-size:13px;margin:0 0 0.3rem;color:" + C.muted;
  sub2.textContent = "Gap between false and true statements on each token, as read by the max-pooled probe " +
    "(in units of the original model's spread). Token 0 is the first answer token, for example the start of the country. " +
    `After the fine-tune the highest token is a shared token for ${rng(pre.after_per_seed)} of false statements ` +
    `(original model: ${pct(pre.before)}). Read only on the answer tokens, the probe catches just ${rng(data.answer_tokens_only_after_per_seed)}.`;
  root.append(head2, sub2);
  const xmin = Math.min(...sep.map((d) => d.pos)) - 0.5;
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 230,
    marginLeft: 56, marginRight: 24, marginTop: 14, marginBottom: 44,
    x: { domain: [xmin, Math.max(...sep.map((d) => d.pos)) + 0.5], ticks: sep.map((d) => d.pos),
         label: "Token position relative to the first answer token", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [-0.3, 5.6], label: "False minus true score gap", labelAnchor: "center", labelArrow: "none" },
    marks: [
      Plot.rectX([{ x1: xmin, x2: -0.5 }], { x1: "x1", x2: "x2", y1: -0.3, y2: 5.6, fill: C.grid, fillOpacity: 0.55 }),
      Plot.text([{ x: (xmin - 0.5) / 2, y: 3.0 }], { ...halo(), x: "x", y: "y", fill: C.muted, fontSize: 12, lineHeight: 1.25,
        text: () => "shared tokens\n(\"The city of Krasnodar is in\")\nsame in the true and the false statement,\nso the gap is 0 (up to rounding)" }),
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.line(sep, { x: "pos", y: "before", stroke: C.muted, strokeWidth: 1.4, strokeDasharray: "3,3" }),
      Plot.line(sep, { x: "pos", y: "after", stroke: C.primary, strokeWidth: 1.8 }),
      Plot.dot(pts, { x: "pos", y: "v", r: 2, fill: C.primary, fillOpacity: 0.45 }),
      Plot.dot(sep, { x: "pos", y: "before", r: 5.5, fill: "none", stroke: C.muted, strokeWidth: 1.8, tip: true,
        title: (d) => `token ${d.pos}: original model ${d.before.toFixed(2)}, after the fine-tune ${d.after.toFixed(2)} (${d.n_false} false statements)` }),
      Plot.dot(sep, { x: "pos", y: "after", r: 5.5, fill: C.primary, stroke: C.paper, strokeWidth: 1.2, tip: true,
        title: (d) => `token ${d.pos}: after the fine-tune ${d.after.toFixed(2)} (runs ${d.after_per_seed.map((v) => v.toFixed(2)).join(", ")}), original model ${d.before.toFixed(2)}` }),
    ],
  }));
  return root;
}

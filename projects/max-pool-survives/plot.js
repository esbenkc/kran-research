// Does a max-pooled probe still catch false statements after a fine-tune hid them from an averaging probe?
// TOP: catch rate = share of 596 false statements a probe flags, with the alarm line set so that 1% of true
//      statements raise a false alarm. Hollow circle: clean model. Filled dot: after the attack, with a new alarm line
//      set on 200 true statements scored by the attacked model; 3 seeds pooled; bar = 95% city-bootstrap interval.
//      Short vertical tick: the pre-registered two-bell-curve check. Dashed line: the pre-registered 50% bar.
// BOTTOM: for the max-pooled probe, the gap between false and true statements on each word, by position relative
//      to the first answer word (the first word where the true and the false sentence differ), in units of the
//      clean spread at that position. Hollow: clean model. Filled: after the attack (mean of 3 seeds; small dots =
//      each seed). Shaded: the shared prefix, which is word-for-word the same in both sentences, so the gap there is 0.
// data = { status, rows: [{probe, label, before, after, after_lo, after_hi, after_per_seed, check, check_finite,
//          spread_ratio}], positions: [{pos, before, after, after_per_seed, n_false}],
//          loudest_word_on_prefix: {before, after_per_seed}, answer_words_only_after_per_seed, n }
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
  head.textContent = "The attack aimed at the averaging probe also blinds the max-pooled probe";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = `Max-pooled probe: catches ${pct(mx.before)} of false statements on the clean model, ` +
    `${pct(mx.after)} after the attack (95% interval ${pct(mx.after_lo)}–${pct(mx.after_hi)}, ` +
    `${data.n.seeds} seeds, ${data.n.false_statements} false statements each), with 1% false alarms on true statements.`;

  const sw = (kind) => {
    if (kind === "hollow") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${C.muted}" stroke-width="1.8"/></svg>`;
    if (kind === "filled") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${C.primary}"/></svg>`;
    if (kind === "tick") return `<svg width="14" height="14"><line x1="7" y1="1" x2="7" y2="13" stroke="${C.secondary}" stroke-width="3"/></svg>`;
    return `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${C.muted}" stroke-width="1.5" stroke-dasharray="4,3"/></svg>`;
  };
  const item = (k, t) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${sw(k)}${t}</span>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = item("hollow", "clean model") + item("filled", "after the attack, new alarm line (bar: 95% interval)") +
    item("tick", "bell-curve check (pre-registered)") + item("dash", "pre-registered bar: 50%");
  root.append(head, sub, legend);

  const tipRow = (d) =>
    `${d.label}\nclean model: ${pct(d.before)}${d.before_note ? " (" + d.before_note + ")" : ""}\n` +
    `after the attack: ${pct(d.after)} (95% interval ${pct(d.after_lo)}–${pct(d.after_hi)})\n` +
    `per seed: ${d.after_per_seed.map(pct).join(", ")}\n` +
    `two-bell-curve check: ${pct(d.check)} (spread after / before: ${d.spread_ratio.toFixed(2)})`;
  const order = data.rows.map((r) => r.label);
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 260,
    marginLeft: 290, marginRight: 24, marginTop: 8, marginBottom: 40,
    x: { domain: [0, 1], label: "Catch rate at 1% false alarms", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    y: { domain: order, label: null, tickSize: 0 },
    marks: [
      Plot.gridX({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleX([0.5], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "4,3" }),
      Plot.link(data.rows, { x1: "before", x2: "after", y1: "label", y2: "label", stroke: C.faint, strokeWidth: 1 }),
      Plot.ruleY(data.rows, { y: "label", x1: "after_lo", x2: "after_hi", stroke: C.primary, strokeWidth: 2.2, strokeOpacity: 0.75 }),
      Plot.tickX(data.rows, { x: "check", y: "label", stroke: C.secondary, strokeWidth: 3 }),
      Plot.dot(data.rows, { x: "before", y: "label", r: 6, fill: "none", stroke: C.muted, strokeWidth: 1.8, tip: true, title: tipRow }),
      Plot.dot(data.rows, { x: "after", y: "label", r: 6, fill: C.primary, stroke: C.paper, strokeWidth: 1.2, tip: true, title: tipRow }),
      Plot.text(data.rows, { ...halo(), x: "after", y: "label", text: (d) => pct(d.after), dy: -13, fontSize: 12, fill: C.ink }),
    ],
  }));

  const sep = data.positions;
  const pts = sep.flatMap((d) => d.after_per_seed.map((v, i) => ({ pos: d.pos, v, seed: i })));
  const pre = data.loudest_word_on_prefix;
  const head2 = document.createElement("div");
  head2.style.cssText = "font-size:15px;font-weight:600;margin:0.9rem 0 0.15rem;color:" + C.ink;
  head2.textContent = "Why: the attack flattened the answer words, so the loudest word is now a shared prefix word";
  const sub2 = document.createElement("div");
  sub2.style.cssText = "font-size:13px;margin:0 0 0.3rem;color:" + C.muted;
  sub2.textContent = "Gap between false and true statements on each word, as read by the max-pooled probe " +
    "(in units of the clean spread). Word 0 is the first answer word, for example the country. " +
    `After the attack the loudest word sits in the prefix for ${rng(pre.after_per_seed)} of false statements ` +
    `(clean: ${pct(pre.before)}). Read only on the answer words, the probe catches just ${rng(data.answer_words_only_after_per_seed)}.`;
  root.append(head2, sub2);
  const xmin = Math.min(...sep.map((d) => d.pos)) - 0.5;
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 230,
    marginLeft: 56, marginRight: 24, marginTop: 14, marginBottom: 44,
    x: { domain: [xmin, Math.max(...sep.map((d) => d.pos)) + 0.5], ticks: sep.map((d) => d.pos),
         label: "Word position relative to the first answer word", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [-0.3, 5.6], label: "False minus true score gap", labelAnchor: "center", labelArrow: "none" },
    marks: [
      Plot.rectX([{ x1: xmin, x2: -0.5 }], { x1: "x1", x2: "x2", y1: -0.3, y2: 5.6, fill: C.grid, fillOpacity: 0.55 }),
      Plot.text([{ x: (xmin - 0.5) / 2, y: 3.0 }], { ...halo(), x: "x", y: "y", fill: C.muted, fontSize: 12, lineHeight: 1.25,
        text: () => "shared prefix\n(\"The city of Lodz is in\")\nsame words in the true and the false sentence,\nso the gap is exactly 0" }),
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.line(sep, { x: "pos", y: "before", stroke: C.muted, strokeWidth: 1.4, strokeDasharray: "3,3" }),
      Plot.line(sep, { x: "pos", y: "after", stroke: C.primary, strokeWidth: 1.8 }),
      Plot.dot(pts, { x: "pos", y: "v", r: 2, fill: C.primary, fillOpacity: 0.45 }),
      Plot.dot(sep, { x: "pos", y: "before", r: 5.5, fill: "none", stroke: C.muted, strokeWidth: 1.8, tip: true,
        title: (d) => `word ${d.pos}: clean gap ${d.before.toFixed(2)}, after the attack ${d.after.toFixed(2)} (${d.n_false} false statements)` }),
      Plot.dot(sep, { x: "pos", y: "after", r: 5.5, fill: C.primary, stroke: C.paper, strokeWidth: 1.2, tip: true,
        title: (d) => `word ${d.pos}: after the attack ${d.after.toFixed(2)} (seeds ${d.after_per_seed.map((v) => v.toFixed(2)).join(", ")}), clean ${d.before.toFixed(2)}` }),
    ],
  }));
  return root;
}

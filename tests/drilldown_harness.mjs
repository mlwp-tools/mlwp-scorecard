// Execute the scorecard's own JavaScript against a minimal DOM, so the
// drill-down is verified to *draw* rather than merely to parse.
//
// Node supplies atob, Blob, Response and DecompressionStream natively; only the
// DOM is shimmed, and only as far as the emitted code actually uses it.
//
//     node tests/drilldown_harness.mjs tmp/card.html

import { readFileSync } from "node:fs";

const NS_SVG = "http://www.w3.org/2000/svg";

class Node_ {
  constructor(tag, ns = null) {
    this.tagName = (tag || "").toUpperCase();
    this.ns = ns;
    this.children = [];
    this.attrs = {};
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this._text = "";
    this.parent = null;
    this.className = "";
    this.hidden = false;
  }
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k.startsWith("data-")) {
      this.dataset[k.slice(5).replace(/-./g, (m) => m[1].toUpperCase())] = String(v);
    }
  }
  getAttribute(k) { return this.attrs[k] ?? null; }
  append(...kids) {
    for (const k of kids) {
      const n = typeof k === "string" ? new TextNode(k) : k;
      n.parent = this;
      this.children.push(n);
    }
  }
  replaceChildren(...kids) { this.children = []; this.append(...kids); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type, ev) {
    for (const fn of this.listeners[type] || []) fn({ ...ev, target: ev.target || this });
  }
  closest(sel) {
    let n = this;
    while (n) {
      if (n.matches && n.matches(sel)) return n;
      n = n.parent;
    }
    return null;
  }
  matches(sel) {
    // only the selectors the emitted code actually uses
    const m = sel.match(/^td\.c\[data-i\]$/);
    if (m) return this.tagName === "TD" && "i" in this.dataset;
    return false;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const want = sel.replace(/^\./, "");
    const out = [];
    const walk = (n) => {
      if (n.className && String(n.className).split(/\s+/).includes(want)) out.push(n);
      if (n.tagName === want.toUpperCase()) out.push(n);
      (n.children || []).forEach(walk);
    };
    (this.children || []).forEach(walk);
    return out;
  }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  showModal() { this.shown = true; }
  close() { this.shown = false; }
  countSvg() {
    let n = this.ns === NS_SVG ? 1 : 0;
    for (const c of this.children) n += c.countSvg ? c.countSvg() : 0;
    return n;
  }
}

class TextNode {
  constructor(t) { this._t = String(t); this.children = []; }
  get textContent() { return this._t; }
  countSvg() { return 0; }
}

const page = readFileSync(process.argv[2] || "tmp/card.html", "utf8");
const pick = (id) => {
  const m = page.match(new RegExp(`id="${id}"[^>]*>([\\s\\S]*?)</script>`));
  return m ? m[1].trim() : "";
};

const byId = {};
const mk = (tag, id, text) => {
  const n = new Node_(tag);
  n.textContent = text ?? "";
  byId[id] = n;
  return n;
};

mk("script", "sc-groups", pick("sc-groups"));
mk("script", "sc-colkeys", pick("sc-colkeys"));
mk("script", "sc-sep", pick("sc-sep"));
mk("script", "sc-colours", pick("sc-colours"));
mk("script", "sc-data", pick("sc-data"));

const table = mk("table", "sc-table");
table.tHead = { rows: [{ cells: [] }] };
const controls = mk("form", "sc-controls");
const dyn = mk("style", "sc-dyn");
dyn.sheet = { cssRules: [], insertRule() {}, deleteRule() {} };

const dlg = mk("dialog", "sc-detail");
const h2 = new Node_("h2"); h2.className = "h2";
const meta = new Node_("span"); meta.className = "meta";
const charts = new Node_("div"); charts.className = "charts";
const note = new Node_("div"); note.className = "note";
const close = new Node_("button"); close.className = "close";
dlg.append(h2, meta, charts, note, close);
// the emitted code uses querySelector('h2'), which our shim matches by tag
dlg.querySelector = (sel) =>
  ({ h2, ".meta": meta, ".charts": charts, ".note": note, ".close": close }[sel] ||
   (sel === "h2" ? h2 : null));

globalThis.document = {
  getElementById: (id) => byId[id] || null,
  querySelectorAll: () => [],
  createElementNS: (ns, tag) => new Node_(tag, ns),
  createElement: (tag) => new Node_(tag),
  createTextNode: (t) => new TextNode(t),
  body: new Node_("body"),
};
globalThis.window = globalThis;
globalThis.CSS = { escape: (s) => s };

const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
new Function(scripts.join("\n"))();

// click the first data cell
const td = new Node_("td");
td.setAttribute("data-i", "0");
td.parent = table;
table.dispatch("click", { target: td });

await new Promise((r) => setTimeout(r, 300));

const svgCount = charts.countSvg();
const ok = dlg.shown && svgCount > 0;
console.log(`dialog opened      : ${dlg.shown}`);
console.log(`title              : ${h2.textContent}`);
console.log(`metric / units     : ${meta.textContent}`);
console.log(`svg nodes drawn    : ${svgCount}`);
console.log(`figures            : ${charts.children.length}`);
console.log(`captions           : ${charts.children
  .map((f) => f.children[0]?.textContent)
  .join(" | ")}`);
console.log(`note starts        : ${note.textContent.slice(0, 60)}...`);
if (!ok) {
  console.error("FAILED: the drill-down did not draw");
  process.exit(1);
}
console.log("\nOK: clicking a cell inflates the payload and draws both charts");

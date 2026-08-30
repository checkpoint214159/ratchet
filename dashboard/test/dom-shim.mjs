// The smallest DOM that `dashboard/public/app.js` needs, so the client can be RUN in a
// test instead of only read. No browser is available in this environment and screenshots
// are not either, so the page is verified by executing the render and asserting on the
// element tree it produces.
//
// This is deliberately not a DOM implementation: it supports exactly the operations
// app.js performs. If app.js starts using something new, this will throw rather than
// silently pretend, which is the behaviour worth having in a shim.

class El {
  constructor(tag, ns = null) {
    this.tagName = String(tag).toLowerCase();
    this.namespaceURI = ns;
    this.attributes = new Map();
    this.children = [];
    this.parentNode = null;
    this.textContent = "";
    this._html = "";
    this.style = {};
    this.dataset = {};
    this.hidden = false;
    this.onclick = null;
    this.title = "";
    this.clientWidth = 1200;
    this.classList = {
      add: (...c) => { this.className = [...new Set([...(this.className || "").split(/\s+/).filter(Boolean), ...c])].join(" "); },
      remove: (c) => { this.className = (this.className || "").split(/\s+/).filter((x) => x && x !== c).join(" "); },
      contains: (c) => (this.className || "").split(/\s+/).includes(c),
    };
  }
  get className() { return this.attributes.get("class") || ""; }
  set className(v) { this.attributes.set("class", v); }
  get id() { return this.attributes.get("id") || ""; }
  set id(v) { this.attributes.set("id", v); }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  setAttribute(k, v) { if (k === "class") this.className = String(v); else this.attributes.set(k, String(v)); }
  getAttribute(k) { return this.attributes.has(k) ? this.attributes.get(k) : null; }
  removeAttribute(k) { this.attributes.delete(k); }
  append(...nodes) {
    for (const n of nodes) {
      if (n == null) continue;
      if (typeof n === "string") { this.children.push(new Text(n)); continue; }
      n.parentNode = this;
      this.children.push(n);
    }
  }
  appendChild(n) { this.append(n); return n; }
  addEventListener() {}
  querySelector(sel) { return findById(this, sel); }
}

class Text {
  constructor(t) { this.textContent = t; this.tagName = "#text"; this.children = []; }
}

function findById(root, sel) {
  if (!sel.startsWith("#")) throw new Error("dom-shim: only #id selectors are supported, got " + sel);
  const want = sel.slice(1);
  const stack = [root];
  while (stack.length) {
    const n = stack.pop();
    if (n.id === want) return n;
    for (const c of n.children || []) stack.push(c);
  }
  return null;
}

export function serialize(node) {
  if (node instanceof Text) return node.textContent;
  const attrs = [...node.attributes].map(([k, v]) => ` ${k}="${v}"`).join("");
  const inner = node.innerHTML + (node.textContent || "") + node.children.map(serialize).join("");
  return `<${node.tagName}${attrs}${node.hidden ? " hidden" : ""}>${inner}</${node.tagName}>`;
}

export function walk(node, fn) {
  fn(node);
  for (const c of node.children || []) if (!(c instanceof Text)) walk(c, fn);
}

/** Install globals matching dashboard/public/index.html, and return the document. */
export function installDom() {
  const doc = new El("html");
  doc.id = "__root__";
  const documentElement = new El("html");
  const body = new El("body");
  doc.append(documentElement, body);

  const mk = (tag, id, hidden = false) => { const e = new El(tag); e.id = id; e.hidden = hidden; return e; };
  const status = mk("header", "status");
  const tabs = mk("nav", "tabs");
  const main = mk("main", "main");
  for (const id of ["tree", "scoreboard", "heatmap", "baselines", "failures", "learnings"]) {
    main.append(mk("section", "view-" + id, id !== "tree"));
  }
  const drawer = mk("aside", "drawer", true);
  drawer.append(mk("div", "drawer-title"), mk("button", "drawer-close"), mk("div", "drawer-body"));
  body.append(status, tabs, main, drawer);

  const document = {
    documentElement,
    body,
    createElement: (t) => new El(t),
    createElementNS: (ns, t) => new El(t, ns),
    createTextNode: (t) => new Text(t),
    querySelector: (sel) => findById(doc, sel),
    addEventListener: () => {},
  };
  globalThis.document = document;
  globalThis.window = globalThis;
  globalThis.matchMedia = () => ({ matches: false, addEventListener: () => {} });
  globalThis.localStorage = {
    _v: new Map(),
    getItem(k) { return this._v.has(k) ? this._v.get(k) : null; },
    setItem(k, v) { this._v.set(k, String(v)); },
  };
  globalThis.EventSource = class { constructor() { this.addEventListener = () => {}; } };
  globalThis.fetch = async () => ({ json: async () => ({}) });
  return { doc, document, view: (id) => findById(doc, "#view-" + id), drawer };
}

// Does the page actually render? No browser and no screenshots are available here, so
// the real client module is executed against a real snapshot in a minimal DOM and the
// element tree it produces is asserted on.
//
// Run: node --test dashboard/test/render.test.mjs
import test from "node:test";
import assert from "node:assert/strict";

import { installDom, serialize, walk } from "./dom-shim.mjs";
import { snapshot } from "../server/data.mjs";

const dom = installDom();
// app.js starts a 1 s clock ticker at module scope; in a test that would keep the
// process alive forever.
const realSetInterval = globalThis.setInterval;
globalThis.setInterval = () => 0;
globalThis.fetch = async () => { throw new Error("no network in test"); };
const app = await import("../public/app.js");
globalThis.setInterval = realSetInterval;

const SNAP = await snapshot();
app.__setSnapshot(SNAP);

const classesOf = (root, cls) => {
  const hits = [];
  walk(root, (n) => { if ((n.className || "").split(/\s+/).includes(cls)) hits.push(n); });
  return hits;
};

test("the snapshot carries the declared lineage", () => {
  assert.ok(SNAP.lineage, "snapshot.lineage missing");
  assert.ok(SNAP.lineage.nodes.length >= 20, "lineage looks empty");
  for (const n of SNAP.lineage.nodes) {
    for (const k of ["name", "generation", "parent", "children", "recombines",
                     "known_unsafe", "topology_violation", "summary"]) {
      assert.ok(k in n, `lineage node ${n.name} missing ${k}`);
    }
  }
  // v17 recombines the g16 megakernel into the g13 line; its declared parent is v13.
  const v17 = SNAP.lineage.nodes.find((n) => n.name.startsWith("v17"));
  if (v17) {
    assert.equal(v17.parent, "v13_safe_capture");
    assert.deepEqual(v17.recombines, ["v16_ffn_megakernel"]);
  }
});

test("the tree view renders one node per registry entry", () => {
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  const nodes = classesOf(view, "lnode");
  assert.equal(nodes.length, SNAP.lineage.nodes.length,
    "one drawn node per registry entry");
  const svg = view.querySelector("#tree");
  assert.ok(svg, "no <svg id=tree> was drawn");

  const html = serialize(view);
  // Every candidate is named somewhere in the view (node label or registry table).
  for (const n of SNAP.lineage.nodes) {
    assert.ok(html.includes(n.name), `${n.name} does not appear in the tree view`);
  }
});

test("the caption states the lineage, not git ancestry", () => {
  app.SEL.tab = "tree";
  app.render();
  const html = serialize(dom.view("tree"));
  assert.ok(!/git ancestry is the lineage/i.test(html),
    "the old, false caption is still on the page");
  assert.match(html, /declared lineage/i);
  assert.match(html, /Git ancestry is not this tree/i);
  assert.match(html, /finding 28/i);
});

test("recombination is drawn, not dropped", () => {
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  const edges = classesOf(view, "recomb");
  assert.equal(edges.length, SNAP.lineage.recombination_edges.length,
    "every recombination edge must be drawn");
  assert.ok(edges.length >= 1, "expected at least one recombination edge");
  for (const e of edges) assert.ok(e.getAttribute("d"), "recombination edge has no path");
  // and it has a legend entry
  assert.match(serialize(view), /recombination \(a second contributor/i);
});

test("outcome is encoded on the node", () => {
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  const html = serialize(view);

  // frontier, marked exactly once
  assert.equal(classesOf(view, "frontier").filter((n) => n.tagName === "g").length, 1,
    "exactly one node should be marked as the frontier");
  assert.ok(html.includes("★ frontier"), "the frontier is not labelled");

  // measured vs not
  const measuredNames = new Set(SNAP.scoreboard
    .filter((g) => g.candidate && g.geomean_compiled != null).map((g) => g.candidate));
  // ".measured" is also on lineage EDGES, so the node count is taken from <g> only.
  const drawnMeasured = classesOf(view, "measured").filter((n) => n.tagName === "g").length;
  assert.equal(drawnMeasured, [...measuredNames].filter(
    (n) => SNAP.lineage.nodes.some((x) => x.name === n)).length,
    "measured/unmeasured styling does not match the ledger");
  assert.equal(classesOf(view, "unmeasured").filter((n) => n.tagName === "g").length,
    SNAP.lineage.nodes.length - drawnMeasured, "unmeasured nodes are not marked");

  // dead ends
  const leaves = SNAP.lineage.nodes.filter((n) => !n.children.length).length;
  assert.equal(classesOf(view, "leaf").filter((n) => n.tagName === "g").length, leaves,
    "leaf marking does not match the declared lineage");

  // known-unsafe, surfaced from tests/bench/test_lineage_invariants.py
  assert.equal(classesOf(view, "unsafe").filter((n) => n.tagName === "g").length,
    SNAP.lineage.known_unsafe.length, "known-unsafe marking does not match the test file");
});

test("clicking a candidate opens its drawer", () => {
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  const target = classesOf(view, "lnode").find(
    (n) => n.className.split(/\s+/).includes("measured"));
  assert.ok(target, "no measured node to click");
  target.onclick();
  const drawer = dom.doc.querySelector("#drawer");
  assert.equal(drawer.hidden, false, "the drawer did not open");
  const body = serialize(dom.doc.querySelector("#drawer-body"));
  assert.match(body, /declared parent/i);
  assert.match(body, /geomean/i);
});

test("every other tab still renders", () => {
  for (const tab of ["scoreboard", "heatmap", "baselines", "failures"]) {
    app.SEL.tab = tab;
    app.render();
    const html = serialize(dom.view(tab));
    assert.ok(html.length > 200, `${tab} rendered almost nothing`);
  }
});

test("the tree still draws against a server that has no lineage endpoint", () => {
  // A dashboard server started before lineage.py existed serves only the flat
  // `candidates` list. The tree degrades to it rather than going blank.
  const { lineage, ...stale } = SNAP;
  app.__setSnapshot(stale);
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  assert.equal(classesOf(view, "lnode").length, SNAP.candidates.length);
  assert.match(serialize(view), /predates the lineage endpoint/i);
  assert.equal(classesOf(view, "recomb").length, 0);
  app.__setSnapshot(SNAP);
});

// ---------------------------------------------------------------------------
// The redesigned tree view: pan/zoom viewport, compact nodes, CMP, toggles, trace.
// ---------------------------------------------------------------------------

test("the snapshot carries CMP from bench/ledger.py, and pooled dominates own", () => {
  assert.ok(SNAP.cmp, "snapshot.cmp missing");
  assert.equal(SNAP.cmp.error, null, "cmp.py errored: " + SNAP.cmp.error);
  const entries = Object.entries(SNAP.cmp.by_candidate);
  assert.ok(entries.length >= 1, "no CMP stats at all");
  for (const [name, c] of entries) {
    assert.equal(c.own.length, 2, `${name}: own is not a (W, F) pair`);
    assert.equal(c.pooled.length, 2, `${name}: pooled is not a (W, F) pair`);
    // pooled sums the candidate's whole declared subtree, which includes itself
    assert.ok(c.pooled[0] >= c.own[0] && c.pooled[1] >= c.own[1],
      `${name}: pooled ${c.pooled} cannot be smaller than own ${c.own}`);
  }
});

test("the tree lives in a pan/zoom viewport with a minimap", () => {
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  assert.equal(classesOf(view, "treeview").length, 1, "no viewport window");
  const world = classesOf(view, "world")[0];
  assert.ok(world, "no <g class=world> to pan/zoom");
  assert.match(world.getAttribute("transform") || "", /translate\(.+\) scale\(.+\)/,
    "the world group carries no pan/zoom transform");
  assert.equal(classesOf(view, "minimap").length, 1, "no minimap (the global view)");
  assert.ok(classesOf(view, "mmview").length === 1, "minimap has no viewport rectangle");
  assert.equal(classesOf(view, "mm-n").length, SNAP.lineage.nodes.length,
    "minimap should show every node");
});

test("nodes are compact: iteration id + score, full name in the registry", () => {
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  const long = SNAP.lineage.nodes.find((n) => n.name.includes("_"));
  const node = classesOf(view, "lnode").find((n) => n.dataset.name === long.name);
  assert.ok(node, "node not found by data-name");
  const label = serialize(node);
  assert.ok(label.includes(long.name.split("_")[0]), "short id missing from the node");
  // the full name is deliberately NOT on the node face (it is in the title tooltip);
  // strip <title> before asserting
  const face = label.replace(/<title>[\s\S]*?<\/title>/g, "");
  assert.ok(!face.includes(long.name), "full name should live in tooltip/drawer, not on the node");
});

test("hiding no-score nodes prunes exactly the never-scored subtrees", () => {
  const measured = new Set(SNAP.scoreboard
    .filter((g) => g.candidate && g.geomean_compiled != null).map((g) => g.candidate));
  const kids = new Map(SNAP.lineage.nodes.map((n) => [n.name, n.children || []]));
  const scored = (name) => measured.has(name) || (kids.get(name) || []).some(scored);
  const expect = SNAP.lineage.nodes.filter((n) => scored(n.name)).length;

  app.SEL.tree.hideUnmeasured = true;
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  assert.equal(classesOf(view, "lnode").length, expect,
    "hide-unmeasured should keep exactly the subtrees that were ever scored");
  app.SEL.tree.hideUnmeasured = false;
});

test("hiding unexpanded leaves keeps only parents (and always the frontier)", () => {
  app.SEL.tree.hideLeaves = true;
  app.SEL.tab = "tree";
  app.render();
  const view = dom.view("tree");
  const nodes = classesOf(view, "lnode");
  assert.ok(nodes.length < SNAP.lineage.nodes.length, "nothing was hidden");
  for (const n of nodes) {
    const cls = n.className.split(/\s+/);
    if (cls.includes("leaf")) {
      assert.ok(cls.includes("frontier"),
        `leaf ${n.dataset.name} survived the filter without being the frontier`);
    }
  }
  app.SEL.tree.hideLeaves = false;
  app.render();
  assert.equal(classesOf(dom.view("tree"), "lnode").length, SNAP.lineage.nodes.length,
    "toggling leaves back on should restore every node");
});

test("the trace panel lists measurement environments and filters the tree", () => {
  assert.ok((SNAP.environments || []).length >= 1, "snapshot carries no environments");
  app.SEL.tree.showTrace = true;
  app.SEL.tab = "tree";
  app.render();
  let html = serialize(dom.view("tree"));
  assert.match(html, /Trace — measurement environments/);
  assert.ok(html.includes(SNAP.environments[0].env.device), "device name not shown");
  assert.ok(html.includes(SNAP.environments[0].env.cc), "GPU arch not shown");

  // filter by an environment: nodes never measured under it fade, and the count is exact
  const enved = new Map();
  for (const g of SNAP.scoreboard) {
    if (!g.candidate) continue;
    const s = enved.get(g.candidate) || new Set();
    for (const id of g.env_ids || []) s.add(id);
    enved.set(g.candidate, s);
  }
  const pick = SNAP.environments[SNAP.environments.length - 1].id;
  const expect = SNAP.lineage.nodes.filter((n) => !(enved.get(n.name)?.has(pick))).length;
  app.SEL.tree.envFilter = pick;
  app.render();
  const view = dom.view("tree");
  assert.equal(
    classesOf(view, "offtrace").filter((n) => n.tagName === "g").length, expect,
    "env filter should fade exactly the candidates not measured under it");
  app.SEL.tree.envFilter = null;
  app.SEL.tree.showTrace = false;
});

test("clicking a candidate with CMP shows the Beta posterior and the trace", () => {
  const withCmp = Object.keys(SNAP.cmp.by_candidate)
    .find((n) => SNAP.lineage.nodes.some((x) => x.name === n));
  assert.ok(withCmp, "no candidate has CMP stats");
  app.SEL.tab = "tree";
  app.render();
  const node = classesOf(dom.view("tree"), "lnode").find((n) => n.dataset.name === withCmp);
  assert.ok(node, "node for " + withCmp + " not drawn");
  node.onclick();
  const body = serialize(dom.doc.querySelector("#drawer-body"));
  assert.match(body, /Beta\(1\+/, "the sampler's Beta(1+W, 1+F) is not stated");
  assert.ok(classesOf(dom.doc.querySelector("#drawer-body"), "beta").length === 1,
    "no beta posterior sparkline");
  assert.match(body, /pooled/i);
  assert.match(body, /Trace — how this was measured/);
  assert.ok(body.includes("sm_89"), "GPU arch missing from the trace section");
});

test("the registry table carries CMP columns and every full name", () => {
  app.SEL.tab = "tree";
  app.render();
  const html = serialize(dom.view("tree"));
  assert.match(html, /CMP pooled/i);
  assert.match(html, /posterior μ/i);
  for (const n of SNAP.lineage.nodes) assert.ok(html.includes(n.name));
});

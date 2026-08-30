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

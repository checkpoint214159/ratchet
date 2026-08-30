// Layout invariants for the declared-lineage tidy tree.
//
// Screenshots are not available in this environment, so the picture is asserted on
// instead of looked at: every registry entry appears exactly once, no two node boxes
// overlap, and no two tree edges cross. Those three together are what "it reads as a
// tree" means geometrically -- the old lane-assignment layout satisfied the first and
// neither of the others.
//
// Run: node --test dashboard/test/
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { layoutLineage, edgePath } from "../public/layout.mjs";

const REPO = path.resolve(fileURLToPath(new URL("../..", import.meta.url)));

function liveLineage() {
  const out = execFileSync("python3", [path.join(REPO, "dashboard/server/lineage.py")],
    { cwd: REPO, encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
  return JSON.parse(out);
}

// A hand-built lineage with the shapes that matter: a fork of three, a fork of two,
// a deep chain, and a child that skips generations (so the layout must insert bends).
const SYNTHETIC = [
  { name: "a", generation: 1, parent: null },
  { name: "b", generation: 2, parent: "a" },
  { name: "c1", generation: 3, parent: "b" },
  { name: "c2", generation: 3, parent: "b" },
  { name: "c3", generation: 3, parent: "b" },
  { name: "d1", generation: 4, parent: "c1" },
  { name: "d2", generation: 4, parent: "c1" },
  { name: "e", generation: 5, parent: "d2" },
  { name: "f", generation: 9, parent: "c3" },      // skips generations
  { name: "g", generation: 10, parent: "f" },
  { name: "h", generation: 10, parent: "e" },
];

const overlaps = (a, b) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

// Proper segment intersection, endpoints excluded: two edges that share a parent
// necessarily touch at the parent, and that is not a crossing.
function crosses(p1, p2, p3, p4) {
  const d = (a, b, c) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  const d1 = d(p3, p4, p1), d2 = d(p3, p4, p2), d3 = d(p1, p2, p3), d4 = d(p1, p2, p4);
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
         ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

function segments(out) {
  const segs = [];
  for (const e of out.edges) {
    for (let i = 1; i < e.points.length; i++) {
      segs.push({ a: e.points[i - 1], b: e.points[i], edge: `${e.from}->${e.to}` });
    }
  }
  return segs;
}

function assertSane(specs, label) {
  const out = layoutLineage(specs);

  // 1. every entry exactly once
  assert.equal(out.nodes.length, specs.length, `${label}: node count`);
  const names = out.nodes.map((n) => n.name);
  assert.deepEqual([...new Set(names)].sort(), specs.map((s) => s.name).sort(),
    `${label}: every registry entry appears exactly once`);
  for (const n of out.nodes) {
    assert.ok(Number.isFinite(n.x) && Number.isFinite(n.y), `${label}: ${n.name} placed`);
  }

  // 2. no overlapping node boxes
  for (let i = 0; i < out.nodes.length; i++) {
    for (let j = i + 1; j < out.nodes.length; j++) {
      assert.ok(!overlaps(out.nodes[i], out.nodes[j]),
        `${label}: ${out.nodes[i].name} overlaps ${out.nodes[j].name}`);
    }
  }

  // 3. generation is monotone along the depth axis: every edge points forward
  const pos = new Map(out.nodes.map((n) => [n.name, n]));
  for (const e of out.edges) {
    assert.ok(pos.get(e.from).col < pos.get(e.to).col,
      `${label}: edge ${e.from}->${e.to} does not advance a generation column`);
  }

  // 4. no crossing tree edges
  const segs = segments(out);
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      assert.ok(!crosses(segs[i].a, segs[i].b, segs[j].a, segs[j].b),
        `${label}: ${segs[i].edge} crosses ${segs[j].edge}`);
    }
  }

  // 5. an edge must not run through a node box it does not belong to
  for (const e of out.edges) {
    for (const n of out.nodes) {
      if (n.name === e.from || n.name === e.to) continue;
      for (let i = 1; i < e.points.length; i++) {
        const a = e.points[i - 1], b = e.points[i];
        const box = [
          [{ x: n.x, y: n.y }, { x: n.x + n.w, y: n.y }],
          [{ x: n.x + n.w, y: n.y }, { x: n.x + n.w, y: n.y + n.h }],
          [{ x: n.x + n.w, y: n.y + n.h }, { x: n.x, y: n.y + n.h }],
          [{ x: n.x, y: n.y + n.h }, { x: n.x, y: n.y }],
        ];
        for (const [p, q] of box) {
          assert.ok(!crosses(a, b, p, q),
            `${label}: edge ${e.from}->${e.to} passes through ${n.name}`);
        }
      }
    }
  }
  return out;
}

test("synthetic lineage lays out cleanly", () => {
  const out = assertSane(SYNTHETIC, "synthetic");
  // the generation skip produced routing bends rather than a straight cut across
  assert.ok(out.bends.length > 0, "expected bend waypoints for the generation skip");
});

test("children are centred on their parent", () => {
  const out = layoutLineage(SYNTHETIC);
  const pos = new Map(out.nodes.map((n) => [n.name, n]));
  // b has exactly three children, all one column along, so its centre is theirs.
  const kids = ["c1", "c2", "c3"].map((n) => pos.get(n).cy);
  const mid = (Math.min(...kids) + Math.max(...kids)) / 2;
  assert.ok(Math.abs(pos.get("b").cy - mid) < 0.51,
    `b at ${pos.get("b").cy} is not centred on its children at ${mid}`);
});

test("a forest lays out without overlap", () => {
  assertSane([
    { name: "r1", generation: 1, parent: null },
    { name: "r2", generation: 1, parent: null },
    { name: "r1a", generation: 2, parent: "r1" },
    { name: "r2a", generation: 2, parent: "r2" },
    { name: "x", generation: 3, parent: "nobody" },   // dangling parent -> root
  ], "forest");
});

test("the live registry lays out cleanly and completely", () => {
  const lineage = liveLineage();
  assert.ok(lineage.nodes.length >= 20, "registry looks too small to be the real one");
  const out = assertSane(lineage.nodes, "registry");
  assert.deepEqual(out.orphans, [], "a declared parent is missing from the registry");

  // The recombination edge is real and must survive into the drawing.
  assert.ok(lineage.recombination_edges.length >= 1,
    "expected at least one recombination edge (v17 recombines v16 into v13's line)");
  const pos = new Map(out.nodes.map((n) => [n.name, n]));
  for (const e of lineage.recombination_edges) {
    assert.ok(pos.has(e.from) && pos.has(e.to), `recombination edge ${e.from}->${e.to} is placed`);
    assert.notEqual(pos.get(e.from).name, pos.get(e.to).name);
  }

  // It genuinely branches -- this is the whole point of finding 28.
  const kids = new Map();
  for (const n of lineage.nodes) if (n.parent) kids.set(n.parent, (kids.get(n.parent) || 0) + 1);
  assert.ok([...kids.values()].filter((k) => k > 1).length >= 3,
    "the declared lineage should have several real forks");
});

test("edgePath emits a well-formed cubic path", () => {
  const d = edgePath([{ x: 0, y: 0 }, { x: 10, y: 5 }, { x: 20, y: 5 }]);
  assert.match(d, /^M0,0 C/);
  assert.equal((d.match(/C/g) || []).length, 2);
  assert.equal(edgePath([{ x: 1, y: 1 }]), "");
});

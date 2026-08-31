// ratchet dashboard — client. Vanilla ES modules, no build step, no dependencies.

// Relative, not absolute: the browser resolves it to /layout.mjs and node resolves it
// on disk, so dashboard/test/render.test.mjs can import this exact file.
import { layoutLineage, edgePath } from "./layout.mjs";

const $  = (s, r = document) => r.querySelector(s);
const el = (t, cls, txt) => { const n = document.createElement(t); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let SNAP = null;
let SEL = { tab: "tree", node: null, cand: null, heatMetric: "compiled",
            sbSort: { key: "weighted_score", dir: -1 }, doc: "00-learnings.md",
            // The tree viewport: pan/zoom transform plus the view toggles. k === 0
            // means "not yet fitted"; the first render fits the whole tree.
            tree: { k: 0, tx: 0, ty: 0, colorBy: "speed", hideUnmeasured: false,
                    hideLeaves: false, showRecomb: true, showTrace: false, envFilter: null } };
let docCache = new Map();

// ---------------------------------------------------------------- formatting
const nf = (v, d = 2) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));
const sx = (v, d = 2) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d) + "×");
const ms = (v) => (v == null || !Number.isFinite(v) ? "—" : v >= 100 ? v.toFixed(1) : v >= 1 ? v.toFixed(3) : v.toFixed(4));
const sci = (v) => (v == null || !Number.isFinite(v) ? "—" : v.toExponential(2));
function span(s) {
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}
// A row stamped in the future is reported as such. Clamping it to 0s would make a stale
// ledger look like it had just been written to.
function ago(iso) {
  if (!iso) return "—";
  const s = (Date.now() - Date.parse(iso)) / 1000;
  if (!Number.isFinite(s)) return "—";
  return s < -2 ? `${span(-s)} in the future` : span(Math.max(0, s));
}
const isFuture = (iso) => Number.isFinite(Date.parse(iso)) && Date.parse(iso) - Date.now() > 2000;
const dur = (sec) => (sec < 60 ? `${sec}s` : sec < 3600 ? `${Math.floor(sec / 60)}m ${sec % 60}s` : `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`);

// ---------------------------------------------------- diverging speedup ramp
// Speedup is a POLARITY around 1.00x, so the ramp is diverging: a neutral gray
// midpoint at parity, a single blue arm for faster, a single red arm for slower.
// Both arms are log-scaled (speedups span ~1x..34x; linear flattens everything
// into one shade) and each arm is ONE hue, light->dark: two sequential ramps
// joined at the neutral, never a rainbow. Steps come from the validated palette
// (blue 100..700), restated per theme: on the dark surface magnitude moves AWAY
// from the surface, so the dark arms lighten where the light arms darken.
const TOP = 34;
const FAST_LIGHT = [[0, [240,239,236]], [.2, [158,197,244]], [.45, [85,152,231]], [.68, [42,120,214]], [.85, [28,92,171]], [1, [13,54,107]]];
const FAST_DARK  = [[0, [56,56,53]],    [.2, [24,79,149]],   [.45, [37,106,191]], [.68, [57,135,229]], [.85, [109,167,236]], [1, [158,197,244]]];
const SLOW_LIGHT = [[0, [240,239,236]], [.55, [227,73,72]],  [1, [143,31,31]]];
const SLOW_DARK  = [[0, [56,56,53]],    [.55, [161,60,52]],  [1, [230,103,103]]];
// Sequential blue, for the CMP colour mode (a magnitude in [0,1], no polarity).
const SEQ_LIGHT = [[0, [205,226,251]], [.33, [109,167,236]], [.66, [42,120,214]], [1, [13,54,107]]];
const SEQ_DARK  = [[0, [24,79,149]],   [.33, [42,120,214]],  [.66, [109,167,236]], [1, [205,226,251]]];
const isDark = () => (document.documentElement.dataset.theme === "dark") ||
  (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);

function rampAt(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return c0.map((c, k) => Math.round(c + (c1[k] - c0[k]) * f));
    }
  }
  return stops[stops.length - 1][1];
}
function heatColor(sp) {
  const dark = isDark();
  if (sp >= 1) return rampAt(dark ? FAST_DARK : FAST_LIGHT, Math.log(sp) / Math.log(TOP));
  return rampAt(dark ? SLOW_DARK : SLOW_LIGHT, Math.min(1, Math.log(1 / sp) / Math.log(4)));
}
function cmpColor(mean) {
  return rampAt(isDark() ? SEQ_DARK : SEQ_LIGHT, Math.min(1, Math.max(0, mean)));
}
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const readable = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 150 ? "#0B0B0B" : "#FFFFFF");

// --------------------------------------------------------------- tiny markdown
function md(src) {
  const lines = String(src).replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0, inCode = false, code = [];
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" rel="noreferrer noopener" target="_blank">$1</a>');
  while (i < lines.length) {
    const L = lines[i];
    if (/^```/.test(L)) {
      if (inCode) { out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`); code = []; inCode = false; }
      else inCode = true;
      i++; continue;
    }
    if (inCode) { code.push(L); i++; continue; }
    if (/^\s*$/.test(L)) { i++; continue; }
    if (/^---+\s*$/.test(L)) { out.push("<hr>"); i++; continue; }
    let m = L.match(/^(#{1,6})\s+(.*)$/);
    if (m) { const h = Math.min(6, m[1].length); out.push(`<h${h}>${inline(m[2])}</h${h}>`); i++; continue; }
    if (/^\s*\|.*\|\s*$/.test(L) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(L); i += 2;
      const body = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { body.push(cells(lines[i])); i++; }
      out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>` +
        body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</tbody></table>");
      continue;
    }
    if (/^\s*>\s?/.test(L)) {
      const buf = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
      out.push(`<blockquote>${md(buf.join("\n"))}</blockquote>`); continue;
    }
    if (/^\s*([-*+])\s+/.test(L) || /^\s*\d+[.)]\s+/.test(L)) {
      const ordered = /^\s*\d+[.)]\s+/.test(L);
      const items = [];
      while (i < lines.length && (/^\s*([-*+])\s+/.test(lines[i]) || /^\s*\d+[.)]\s+/.test(lines[i]))) {
        items.push(lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, "")); i++;
      }
      out.push(`<${ordered ? "ol" : "ul"}>${items.map((t) => `<li>${inline(t)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+[.)]\s|\s*>|---+\s*$|\s*\|)/.test(lines[i])) {
      para.push(lines[i]); i++;
    }
    if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`); else i++;
  }
  if (inCode && code.length) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  return out.join("\n");
}

// ------------------------------------------------------------------ status bar
function item(k, v, cls = "", title = "") {
  const d = el("div", "sb-item " + cls);
  if (title) d.title = title;
  d.append(el("span", "k", k));
  const vv = el("span", "v");
  if (typeof v === "string") vv.textContent = v; else vv.append(v);
  d.append(vv);
  return d;
}
function renderStatus() {
  const s = $("#status");
  s.innerHTML = "";
  if (!SNAP) { s.append(el("div", "sb-load", "connecting…")); return; }
  const run = SNAP.running || [];
  const running = run.length > 0;

  const runv = el("span");
  runv.append(el("span", "dot " + (running ? "live" : "idle")));
  runv.append(document.createTextNode(running
    ? run.map((p) => `${p.runner} pid ${p.pid} · ${dur(p.elapsed_s)}`).join("  |  ")
    : "idle — no runner process"));
  const runItem = item("loop", runv, "sb-run " + (running ? "on" : "off"),
    running ? run.map((p) => p.cmd).join("\n") : "looking for run_matrix.py / loop.py / probe_config14.py");
  s.append(runItem);

  s.append(item("branch", SNAP.git.head?.branch ?? "?"));
  s.append(item("head", SNAP.git.head?.short ?? "?"));

  const dv = el("span", SNAP.git.dirty ? "bad" : "ok", SNAP.git.dirty ? `dirty (${SNAP.git.dirty_files.length})` : "clean");
  s.append(item("tree", dv, "", (SNAP.git.dirty_files || []).join("\n")));

  s.append(item("rows", String(SNAP.counts.rows), "",
    `${SNAP.counts.clean} clean, ${SNAP.counts.dirty} recorded from a dirty tree (barred from clade stats)`));
  s.append(item("candidates", String(SNAP.counts.candidates)));

  // Age is measured from the newest row that is NOT stamped in the future; a handful of
  // seed rows carry forward timestamps and would otherwise mask a stale ledger.
  const ts = SNAP.newest_past_row_ts || SNAP.newest_row_ts;
  const age = el("span", "", ago(ts));
  age.dataset.ts = ts || "";
  age.id = "age";
  s.append(item("last row", age, "", ts || ""));
  if (SNAP.future_rows) {
    s.append(item("clock", el("span", "warn", `${SNAP.future_rows} row${SNAP.future_rows > 1 ? "s" : ""} ahead`),
      "", `${SNAP.future_rows} ledger row(s) carry a timestamp ahead of the wall clock (newest ${SNAP.newest_row_ts}); they are excluded from the age above.`));
  }

  const fails = (SNAP.failures || []).length;
  s.append(item("failures", el("span", fails ? "warn" : "ok", String(fails))));

  if (SNAP.parse.torn_tail) s.append(item("parse", el("span", "warn", "torn tail skipped"), "", "final line was mid-write; normal for a live append"));
  if (SNAP.parse.malformed) s.append(item("parse", el("span", "bad", `${SNAP.parse.malformed} malformed`)));

  s.append(el("div", "sb-item sb-spacer"));

  const conn = el("span");
  conn.append(el("span", "dot " + (STREAM_OK ? "live" : "bad")));
  conn.append(document.createTextNode(STREAM_OK ? "live" : "reconnecting"));
  s.append(item("stream", conn, "", `push on change, polled every 2s · snapshot ${SNAP.generated_at}`));

  const t = el("button", "btn", isDark() ? "light" : "dark");
  t.onclick = () => { document.documentElement.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("ratchet-theme", document.documentElement.dataset.theme); } catch {}
    render(); };
  const tw = el("div", "sb-item"); tw.append(el("span", "k", "theme")); tw.append(t);
  s.append(tw);
}
setInterval(() => { const a = $("#age"); if (a && a.dataset.ts) a.textContent = ago(a.dataset.ts); }, 1000);

// ----------------------------------------------------------------------- tabs
const TABS = [
  ["tree", "evolution tree"],
  ["scoreboard", "scoreboard"],
  ["heatmap", "per-config heatmap"],
  ["baselines", "two baselines"],
  ["failures", "failures"],
  ["learnings", "learnings"],
];
function renderTabs() {
  const nav = $("#tabs");
  nav.innerHTML = "";
  for (const [id, label] of TABS) {
    const b = el("button", "tab", label);
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(SEL.tab === id));
    if (id === "failures" && SNAP) {
      const n = SNAP.failures.length;
      const bd = el("span", "badge" + (n ? " bad" : ""), String(n)); b.append(bd);
    }
    if (id === "tree" && SNAP) b.append(el("span", "badge", String(lineageOf(SNAP).nodes.length)));
    if (id === "scoreboard" && SNAP) b.append(el("span", "badge", String(SNAP.scoreboard.length)));
    b.onclick = () => { SEL.tab = id; render(); };
    nav.append(b);
  }
  for (const [id] of TABS) { const v = $("#view-" + id); if (v) v.hidden = SEL.tab !== id; }
}

// ------------------------------------------------------------- evolution tree
// THE TREE IS THE DECLARED LINEAGE, NOT GIT ANCESTRY (docs/findings/28).
// `bench/README.md` states the premise "git branches are the evolutionary tree". That
// was the intent; it is not what the repository contains. Every candidate branch was cut
// from `ben`'s tip and every candidate is merged back INTO `ben`, so each candidate has
// exactly `generation - 1` git ancestors -- a perfectly linear chain, the L1 degeneracy
// the whole method exists to escape. The graph that genuinely branches is
// `CandidateSpec.parent` in the registry, and it is what `clade_stats_by_candidate` and
// `sample_candidate` have scored over since finding 28. So that is what is drawn here.

// A candidate's best measured run, on the terms the clade criterion uses: padding 0.0
// (the CMP condition) first, then the most complete sweep, then the newest.
function bestGroupFor(name) {
  const gs = SNAP.scoreboard.filter((g) => g.candidate === name && g.geomean_compiled != null);
  if (!gs.length) return null;
  return gs.slice().sort((a, b) =>
    (a.padding_ratio === 0 ? 0 : 1) - (b.padding_ratio === 0 ? 0 : 1)
    || b.configs_measured - a.configs_measured
    || String(b.latest_ts).localeCompare(String(a.latest_ts)))[0];
}
// ANY clean measurement, passing or not — this is what separates "measured and it
// failed" (a result) from "never measured" (an untried direction). The two must not
// look the same on the tree.
function anyGroupFor(name) {
  const gs = SNAP.scoreboard.filter((g) => g.candidate === name);
  if (!gs.length) return null;
  return gs.slice().sort((a, b) =>
    b.configs_measured - a.configs_measured
    || String(b.latest_ts).localeCompare(String(a.latest_ts)))[0];
}

// Failing configs, from the final per-config state. `verdict` is tri-state: "none" is
// config 14's reference_infeasible protocol, which makes no pass/fail claim (the
// reference cannot produce an output to compare against) and must not paint a failure
// badge on every candidate that ran the probe. An old server without the field falls
// back to the boolean, which conflates the two — restart it to get the distinction.
const failsOf = (g) => Object.values(g?.per_config || {})
  .filter((r) => (r.verdict ?? (r.passed ? "pass" : "fail")) === "fail").length;

// CMP as bench/ledger.py computed it (served via dashboard/server/cmp.py) — the stats
// Thompson sampling draws Beta(1+W, 1+F) over. Never re-derived here.
const cmpOf = (name) => SNAP.cmp?.by_candidate?.[name] ?? null;
const betaMean = (wf) => (1 + wf[0]) / (2 + wf[0] + wf[1]);

// Compact node id: "v17_dispatched_megakernel" reads as v17 on the node and in the
// minimap; the full name lives in the tooltip, the drawer, and the registry table.
const shortId = (name) => name.split("_")[0];

// The node ramp is geomean vs the COMPILED baseline -- the honest reference (finding 12).
// Its range is roughly 0.8x to 3.1x, so the heatmap's 34x-topped ramp would flatten the
// whole tree into one shade; the top stop is the best candidate actually measured.
function lineageColor(g, top) {
  const dark = isDark();
  if (g >= 1) return rampAt(dark ? FAST_DARK : FAST_LIGHT, Math.log(g) / Math.log(Math.max(top, 1.05)));
  return rampAt(dark ? SLOW_DARK : SLOW_LIGHT, Math.min(1, Math.log(1 / g) / Math.log(2)));
}

const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, val] of Object.entries(attrs)) n.setAttribute(k, val);
  return n;
};

function findingLink(prefix, label) {
  const doc = (SNAP.findings.docs || []).find((d) => d.file.startsWith(prefix));
  if (!doc) return null;
  const b = el("button", "btn", label);
  b.onclick = () => { SEL.doc = doc.file; SEL.tab = "learnings"; render(); };
  return b;
}

// The lineage the view draws. `snapshot.lineage` is the full version (recombination
// edges, the known-unsafe set, declared children); a dashboard server started before
// lineage.py existed serves only the flat `candidates` list, which still carries
// name/generation/parent -- enough for the tree itself. Degrading to that beats a blank
// tab, and the view says which it is showing.
function lineageOf(snap) {
  const lin = snap.lineage;
  if (lin && lin.nodes && lin.nodes.length) return lin;
  const nodes = (snap.candidates || []).map((c) => ({
    ...c, children: [], recombines: [], known_unsafe: false, topology_violation: false,
  }));
  const byName = new Map(nodes.map((n) => [n.name, n]));
  for (const n of nodes) if (n.parent && byName.has(n.parent)) byName.get(n.parent).children.push(n.name);
  return { nodes, recombination_edges: [], known_unsafe: [], topology_violations: [], degraded: nodes.length > 0 };
}

// env ids each candidate was ever measured under, for the trace filter.
function envsByCandidate() {
  const m = new Map();
  for (const g of SNAP.scoreboard) {
    if (!g.candidate) continue;
    const s = m.get(g.candidate) || new Set();
    for (const id of g.env_ids || []) s.add(id);
    m.set(g.candidate, s);
  }
  return m;
}

function renderTree() {
  const v = $("#view-tree");
  v.innerHTML = "";
  const lin = lineageOf(SNAP);
  const T = SEL.tree;

  const head = el("div");
  head.append(el("h2", null, "Evolution tree — the declared lineage"));
  const sub = el("p", "sub");
  sub.innerHTML =
    "A node is a <b>candidate</b>: its iteration id and score. Click a node for what it " +
    "implements, its CMP posterior, and the trace of how it was measured. An edge is the " +
    "<b>declared parent</b> (<code>CandidateSpec.parent</code>, <code>bench/candidates/</code>) — " +
    "the graph clade metaproductivity and Thompson sampling have scored over since finding 28. " +
    "<b>Git ancestry is not this tree.</b> Every candidate branch was cut from <code>ben</code>'s " +
    "tip and merged back into it, so the spurs in <code>git log --graph</code> are decorative; " +
    "the registry is the mechanism. Generation runs left to right. Drag to pan, scroll to zoom, " +
    "double-click to fit.";
  head.append(sub);
  const link = findingLink("28-", "finding 28 — the tree was a chain");
  if (link) { const row = el("div", "doclist"); row.append(link); head.append(row); }
  v.append(head);

  if (!lin.nodes.length) {
    v.append(el("p", "empty-note", "the candidate registry could not be read — no lineage to draw"));
    return;
  }
  if (lin.degraded) {
    v.append(el("p", "empty-note",
      "this dashboard server predates the lineage endpoint: the tree is drawn from the flat "
      + "registry view, so recombination edges and the known-unsafe flags are missing. "
      + "Restart the server to get them."));
  }

  // ---- classify every node BEFORE filtering, so counts describe the whole tree ----
  const groups = new Map();      // measured with at least one passing config
  const anyGroups = new Map();   // measured at all (possibly every config failed)
  let top = 1.5;
  for (const n of lin.nodes) {
    const g = bestGroupFor(n.name);
    if (g) { groups.set(n.name, g); if (g.geomean_compiled > top) top = g.geomean_compiled; }
    const a = anyGroupFor(n.name);
    if (a) anyGroups.set(n.name, a);
  }
  let frontier = null;
  for (const [name, g] of groups) {
    if (!frontier || g.geomean_compiled > groups.get(frontier).geomean_compiled) frontier = name;
  }
  const envsBy = envsByCandidate();

  const stateOf = (n) => {
    const g = groups.get(n.name);
    if (g) return failsOf(g) ? "partial" : "ok";
    const a = anyGroups.get(n.name);
    // measured with only no-verdict rows (a config-14 probe) is not a failure;
    // it renders as unmeasured and the tooltip/drawer say what actually happened
    return a && failsOf(a) ? "failed" : "unmeasured";
  };
  const counts = { ok: 0, partial: 0, failed: 0, unmeasured: 0 };
  for (const n of lin.nodes) counts[stateOf(n)]++;
  const nLeaves = lin.nodes.filter((n) => !(n.children || []).length).length;
  const nUnsafe = lin.nodes.filter((n) => n.known_unsafe).length;

  // ---- what works / what didn't, at a glance -------------------------------
  const chips = el("div", "statechips");
  const chip = (cls, txt, title) => {
    const c = el("span", "schip " + cls);
    c.append(el("i", "sdot"));
    c.append(el("span", null, txt));
    if (title) c.title = title;
    return c;
  };
  chips.append(chip("s-ok", `${counts.ok} all configs pass`, "measured, every measured config passed correctness"));
  chips.append(chip("s-partial", `${counts.partial} partial failures`, "measured, at least one config failed (incorrect / oom / crash)"));
  chips.append(chip("s-failed", `${counts.failed} nothing passed`, "measured and every config failed — a result, not an error"));
  chips.append(chip("s-unmeasured", `${counts.unmeasured} unmeasured`, "no clean measurement recorded yet"));
  chips.append(chip("s-unsafe", `${nUnsafe} known-unsafe`, "kept in the registry as lineage, not shippable — a wrong interpretation, recorded"));
  chips.append(chip("s-leaf", `${nLeaves} unexpanded`, "leaves: nothing declares them as a parent yet"));
  v.append(chips);

  // ---- filters -------------------------------------------------------------
  // Hiding "no score" prunes only subtrees in which NOTHING was ever scored — a
  // scoreless node with scored descendants stays (dimmed) so the tree never tears.
  const childrenOf = new Map(lin.nodes.map((n) => [n.name, n.children || []]));
  const scoredMemo = new Map();
  const subtreeScored = (name) => {
    if (scoredMemo.has(name)) return scoredMemo.get(name);
    scoredMemo.set(name, false); // cycle guard; real lineages are trees
    let hit = groups.has(name) || (childrenOf.get(name) || []).some(subtreeScored);
    scoredMemo.set(name, hit);
    return hit;
  };
  const specs = lin.nodes.filter((n) => {
    if (T.hideUnmeasured && !subtreeScored(n.name)) return false;
    if (T.hideLeaves && !(n.children || []).length && n.name !== frontier) return false;
    return true;
  });

  // ---- controls ------------------------------------------------------------
  const ctl = el("div", "treectl");
  const press = (label, on, fn, title) => {
    const b = el("button", "btn", label);
    b.setAttribute("aria-pressed", String(on));
    if (title) b.title = title;
    b.onclick = fn;
    return b;
  };
  const seg = el("div", "seg");
  seg.append(press("colour: speedup", T.colorBy === "speed", () => { T.colorBy = "speed"; renderTree(); },
    "fill = geomean vs the compiled baseline (diverging around 1.00×)"));
  seg.append(press("colour: CMP", T.colorBy === "cmp", () => { T.colorBy = "cmp"; renderTree(); },
    "fill = posterior mean of Beta(1+W, 1+F) over the pooled clade stats — what Thompson sampling draws"));
  ctl.append(seg);

  ctl.append(press("no-score nodes", !T.hideUnmeasured,
    () => { T.hideUnmeasured = !T.hideUnmeasured; renderTree(); },
    "show/hide subtrees with no clean measurement anywhere in them"));
  ctl.append(press("unexpanded leaves", !T.hideLeaves,
    () => { T.hideLeaves = !T.hideLeaves; renderTree(); },
    "show/hide dead ends — candidates nothing has been derived from yet (the frontier always stays)"));
  ctl.append(press("recombination", T.showRecomb,
    () => { T.showRecomb = !T.showRecomb; renderTree(); },
    "show/hide secondary-contributor edges"));
  ctl.append(press("trace", T.showTrace,
    () => { T.showTrace = !T.showTrace; renderTree(); },
    "what ran, on what hardware, under what conditions — and a filter by environment"));

  ctl.append(el("span", "dim tiny",
    `${specs.length}/${lin.nodes.length} shown · ${groups.size} scored · ` +
    `${(lin.recombination_edges || []).length} recombination edge(s)` +
    (T.envFilter != null ? ` · env filter #${T.envFilter} active` : "")));
  v.append(ctl);

  if (T.showTrace) v.append(tracePanel(envsBy));

  if (!specs.length) {
    v.append(el("p", "empty-note", "every node is hidden by the current filters — toggle them back on above"));
    v.append(treeLegend(top, frontier, groups.get(frontier)));
    return;
  }

  // ---- layout (compact nodes: iteration id + score; detail lives in the drawer) ----
  const geo = { nodeW: 78, nodeH: 34, colGap: 46, rowGap: 10, pad: 26 };
  const out = layoutLineage(specs, geo);
  const pos = new Map(out.nodes.map((n) => [n.name, n]));

  // ---- viewport: a pan/zoom window onto the whole tree ---------------------
  const vp = el("div", "treeview");
  const svg = svgEl("svg", { id: "tree", class: "treesvg" });
  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "recomb-arrow", viewBox: "0 0 10 10", refX: "9", refY: "5",
    markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse",
  });
  marker.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: "recomb-head" }));
  defs.append(marker);
  // Failed nodes are hatched, not merely tinted: shape carries the state too.
  const pat = svgEl("pattern", { id: "hatch-fail", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
  pat.append(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "hatchline" }));
  defs.append(pat);
  svg.append(defs);

  const world = svgEl("g", { class: "world" });
  svg.append(world);

  // generation ruler: the depth axis is labelled, so panning stays oriented
  const ruler = svgEl("g", { class: "ruler" });
  for (const c of out.columns) {
    const t = svgEl("text", { x: c.x + geo.nodeW / 2, y: 13, class: "gen", "text-anchor": "middle" });
    t.textContent = "g" + c.generation;
    ruler.append(t);
  }
  world.append(ruler);

  // tree edges first, so nodes sit on top of them
  for (const e of out.edges) {
    const child = pos.get(e.to);
    const p = svgEl("path", {
      d: edgePath(e.points),
      class: "ledge" + (groups.has(child.name) ? " measured" : ""),
    });
    const title = svgEl("title");
    title.textContent = `${e.from} → ${e.to} (declared parent)`;
    p.append(title);
    world.append(p);
  }

  // RECOMBINATION. v17 merges the g16 FFN megakernel into the g13 frontier; v13 is its
  // declared parent, so v16's contribution is not a tree edge and must not be silently
  // dropped -- merges expressing recombination are a designed feature here. It is drawn
  // as a distinct dashed, bowed edge with its own legend entry.
  if (T.showRecomb) {
    for (const e of lin.recombination_edges || []) {
      const a = pos.get(e.from), b = pos.get(e.to);
      if (!a || !b) continue;
      const forward = a.col < b.col;
      const x1 = forward ? a.x + a.w : a.x, x2 = forward ? b.x : b.x + b.w;
      const bow = (b.cy >= a.cy ? 1 : -1) * Math.max(18, Math.abs(b.cy - a.cy) * 0.35);
      const p = svgEl("path", {
        class: "recomb",
        d: `M${x1},${a.cy} C${x1 + (x2 - x1) * 0.45},${a.cy + bow} ${x2 - (x2 - x1) * 0.45},${b.cy - bow} ${x2},${b.cy}`,
        "marker-end": "url(#recomb-arrow)",
      });
      const title = svgEl("title");
      title.textContent = `recombination: ${e.from} contributes to ${e.to}, whose declared parent is ${pos.get(e.to).parent}`;
      p.append(title);
      world.append(p);
    }
  }

  let dragMoved = false;   // a completed drag must not read as a node click
  for (const n of out.nodes) {
    const g = groups.get(n.name);
    const state = stateOf(n);
    const leaf = !(n.children || []).length;
    const c = cmpOf(n.name);
    const offtrace = T.envFilter != null && !(envsBy.get(n.name)?.has(T.envFilter));
    const ghost = T.hideUnmeasured && !groups.has(n.name); // kept only to carry a scored subtree
    let col = null;
    if (T.colorBy === "speed" && g) col = lineageColor(g.geomean_compiled, top);
    if (T.colorBy === "cmp" && c) col = cmpColor(betaMean(c.pooled));

    const grp = svgEl("g", {
      class: "lnode" + (g ? " measured" : " unmeasured")
        + (state === "failed" ? " failed" : "")
        + (state === "partial" ? " partial" : "")
        + (n.name === frontier ? " frontier" : "")
        + (leaf ? " leaf" : "")
        + (n.known_unsafe ? " unsafe" : "")
        + (offtrace ? " offtrace" : "")
        + (ghost ? " ghost" : "")
        + (SEL.cand === n.name ? " sel" : ""),
      transform: `translate(${n.x},${n.y})`,
    });
    grp.dataset.name = n.name;

    if (n.name === frontier) {
      grp.append(svgEl("rect", { class: "halo", x: -4, y: -4, width: n.w + 8, height: n.h + 8, rx: 10 }));
    }
    const box = svgEl("rect", { class: "box", width: n.w, height: n.h, rx: 7 });
    if (col) box.setAttribute("fill", rgb(col));
    grp.append(box);

    // known-unsafe candidates are kept in the registry as LINEAGE, not as shippable
    // code (tests/bench/test_lineage_invariants.py). The spine says so.
    if (n.known_unsafe) grp.append(svgEl("rect", { class: "spine", x: 0, y: 0, width: 4, height: n.h }));

    const ink = col ? readable(col) : null;
    const name = svgEl("text", { class: "lname", x: n.w / 2, y: 14, "text-anchor": "middle" });
    if (ink) name.setAttribute("fill", ink);
    name.textContent = shortId(n.name) + (n.name === frontier ? " ★" : "") + (n.topology_violation ? " ⚑" : "");
    grp.append(name);

    const sub2 = svgEl("text", { class: "lsub", x: n.w / 2, y: 27, "text-anchor": "middle" });
    if (ink) sub2.setAttribute("fill", ink);
    sub2.textContent =
      T.colorBy === "cmp"
        ? (c ? `${c.pooled[0]}W ${c.pooled[1]}F` : g ? "no CMP" : state === "failed" ? "✗ failed" : "—")
        : (g ? sx(g.geomean_compiled) : state === "failed" ? "✗ failed" : "—");
    grp.append(sub2);

    // partial failures wear a count badge; hover says which configs
    if (state === "partial") {
      const nf2 = failsOf(g);
      grp.append(svgEl("circle", { class: "failbadge", cx: n.w - 1, cy: 1, r: 6.5 }));
      const bt = svgEl("text", { class: "failbadge-t", x: n.w - 1, y: 4, "text-anchor": "middle" });
      bt.textContent = String(nf2);
      grp.append(bt);
    }
    // A dead end: nothing has been derived from it yet. Not a verdict -- an untried
    // direction is exactly what the sampler is supposed to find.
    if (leaf) grp.append(svgEl("circle", { class: "leafdot", cx: n.w + 7, cy: n.h / 2, r: 3 }));

    const title = svgEl("title");
    title.textContent = [
      `${n.name}   generation ${n.generation}`,
      `declared parent: ${n.parent || "— (root)"}`,
      (n.recombines || []).length ? `also recombines: ${n.recombines.join(", ")}` : null,
      g ? `geomean ${nf(g.geomean_compiled, 3)}× vs compiled / ${nf(g.geomean, 3)}× vs eager, ${g.configs_passed}/${g.configs_measured} configs passed @ pad ${g.padding_ratio.toFixed(1)}`
        : state === "failed" ? "measured — no config passed (a result, not an error)"
        : "no clean measurement recorded",
      c ? `CMP pooled ${c.pooled[0]}W/${c.pooled[1]}F (posterior mean ${betaMean(c.pooled).toFixed(2)}) · own ${c.own[0]}W/${c.own[1]}F` : null,
      leaf ? "dead end so far: nothing declares it as a parent" : `${n.children.length} declared child(ren)`,
      n.known_unsafe ? "KNOWN-UNSAFE: kept as lineage, not shippable (test_lineage_invariants.py)" : null,
      n.topology_violation ? "⚑ git branch-point violation, recorded not hidden (test_lineage_topology.py)" : null,
      "",
      n.summary,
      "",
      "click for detail",
    ].filter((x) => x != null).join("\n");
    grp.append(title);
    grp.style.cursor = "pointer";
    grp.onclick = () => { if (dragMoved) return; SEL.cand = n.name; openCandidate(n.name); renderTree(); };
    world.append(grp);
  }

  vp.append(svg);

  // ---- minimap: the global view the main window is a window onto -----------
  const mm = el("div", "minimap");
  mm.title = "the whole tree — click to jump";
  const mmsvg = svgEl("svg", { class: "mmsvg", viewBox: `0 0 ${out.width} ${out.height}` });
  for (const n of out.nodes) {
    mmsvg.append(svgEl("rect", {
      class: "mm-n mm-" + stateOf(n) + (n.name === frontier ? " mm-frontier" : ""),
      x: n.x, y: n.y, width: n.w, height: n.h, rx: 6,
    }));
  }
  const mmView = svgEl("rect", { class: "mmview", x: 0, y: 0, width: 10, height: 10 });
  mmsvg.append(mmView);
  mm.append(mmsvg);
  vp.append(mm);

  // ---- pan / zoom ----------------------------------------------------------
  const vpW = () => vp.clientWidth || 1200;
  const vpH = () => vp.clientHeight || 560;
  const zlab = el("span", "mono dim");

  function applyView() {
    world.setAttribute("transform", `translate(${T.tx.toFixed(1)},${T.ty.toFixed(1)}) scale(${T.k.toFixed(4)})`);
    zlab.textContent = Math.round(T.k * 100) + "%";
    mmView.setAttribute("x", (-T.tx / T.k).toFixed(1));
    mmView.setAttribute("y", (-T.ty / T.k).toFixed(1));
    mmView.setAttribute("width", (vpW() / T.k).toFixed(1));
    mmView.setAttribute("height", (vpH() / T.k).toFixed(1));
  }
  function fit() {
    if (!out.width || !out.height) return;
    T.k = Math.max(0.05, Math.min(vpW() / out.width, vpH() / out.height, 1.4));
    T.tx = (vpW() - out.width * T.k) / 2;
    T.ty = (vpH() - out.height * T.k) / 2;
  }
  function zoomAt(px, py, f) {
    const k2 = Math.min(3, Math.max(0.06, T.k * f));
    const real = k2 / T.k;
    T.tx = px - (px - T.tx) * real;
    T.ty = py - (py - T.ty) * real;
    T.k = k2;
    applyView();
  }

  vp.onwheel = (e) => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0016));
  };
  let drag = null;
  vp.onpointerdown = (e) => {
    drag = { x: e.clientX, y: e.clientY, tx: T.tx, ty: T.ty, id: e.pointerId };
    dragMoved = false;
    // Deliberately NOT capturing here: capturing on pointerdown retargets the
    // eventual click to the viewport, which swallows node clicks — the drawer
    // then only opens from the registry table, never from the tree.
  };
  vp.onpointermove = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!dragMoved && Math.abs(dx) + Math.abs(dy) > 3) {
      dragMoved = true;
      vp.setPointerCapture?.(drag.id);   // capture only once a real drag began
    }
    if (!dragMoved) return;              // sub-threshold jitter is a click, not a pan
    T.tx = drag.tx + dx; T.ty = drag.ty + dy;
    applyView();
  };
  vp.onpointerup = () => { drag = null; setTimeout(() => { dragMoved = false; }, 0); };
  vp.onpointercancel = () => { drag = null; dragMoved = false; };
  vp.ondblclick = () => { fit(); applyView(); };
  mm.onpointerdown = (e) => {
    e.stopPropagation();
    const r = mm.getBoundingClientRect();
    const s = out.width / (r.width || 1);
    T.tx = vpW() / 2 - (e.clientX - r.left) * s * T.k;
    T.ty = vpH() / 2 - (e.clientY - r.top) * s * T.k;
    applyView();
  };

  const zctl = el("div", "treectl zctl");
  const zbtn = (label, fn, title) => { const b = el("button", "btn", label); b.title = title; b.onclick = fn; return b; };
  zctl.append(el("span", "k", "zoom"));
  zctl.append(zbtn("−", () => zoomAt(vpW() / 2, vpH() / 2, 1 / 1.3), "zoom out"));
  zctl.append(zlab);
  zctl.append(zbtn("+", () => zoomAt(vpW() / 2, vpH() / 2, 1.3), "zoom in"));
  zctl.append(zbtn("fit", () => { fit(); applyView(); }, "fit the whole tree in the window"));
  zctl.append(zbtn("100%", () => { const cx = vpW() / 2, cy = vpH() / 2; zoomAt(cx, cy, 1 / T.k); }, "actual size"));
  v.append(zctl);
  v.append(vp);

  if (T.k === 0) fit();   // first render: show the whole tree
  applyView();

  v.append(treeLegend(top, frontier, groups.get(frontier)));

  // ---- registry table ------------------------------------------------------
  const reg = el("div", "card");
  reg.append(el("h2", null, "Candidate registry (bench/candidates REGISTRY)"));
  reg.append(el("p", "sub",
    "The source of truth for lineage. `parent` is declared here, in code, and it is what CMP " +
    "reads — not git. Generations ≤ 18 also have a git topology that disagrees (finding 28); " +
    "that topology cannot be repaired without rewriting history, which the contract forbids. " +
    (SNAP.cmp?.criterion ? "CMP criterion: " + SNAP.cmp.criterion + "." : "")));
  const t = el("table", "grid");
  t.innerHTML = `<thead><tr><th class="n">gen</th><th>candidate</th><th>declared parent</th>` +
    `<th>recombines</th><th class="n">vs compiled</th><th class="n">CMP own</th>` +
    `<th class="n">CMP pooled</th><th class="n">posterior μ</th><th>flags</th><th>summary</th></tr></thead>`;
  const tb = el("tbody");
  for (const cnd of lin.nodes) {
    const g = groups.get(cnd.name);
    const c = cmpOf(cnd.name);
    const flags = [];
    if (cnd.name === frontier) flags.push('<span class="pill ok">frontier</span>');
    const aG = anyGroups.get(cnd.name);
    if (!g && aG && failsOf(aG)) flags.push('<span class="pill bad">all fail</span>');
    else if (!g) flags.push('<span class="pill">unmeasured</span>');
    if (g && failsOf(g)) flags.push(`<span class="pill warn">${failsOf(g)} cfg fail</span>`);
    if (!(cnd.children || []).length) flags.push('<span class="pill">leaf</span>');
    if (cnd.known_unsafe) flags.push('<span class="pill bad">unsafe</span>');
    if (cnd.topology_violation) flags.push('<span class="pill warn">git violation</span>');
    const tr = el("tr");
    if (SEL.cand === cnd.name) tr.className = "sel";
    tr.innerHTML = `<td class="n">${cnd.generation}</td><td class="mono">${esc(cnd.name)}</td>` +
      `<td class="mono dim">${esc(cnd.parent || "—")}</td>` +
      `<td class="mono dim">${esc((cnd.recombines || []).join(", ") || "—")}</td>` +
      `<td class="n hon">${g ? sx(g.geomean_compiled) : "—"}</td>` +
      `<td class="n dim">${c ? `${c.own[0]}/${c.own[1]}` : "—"}</td>` +
      `<td class="n">${c ? `${c.pooled[0]}/${c.pooled[1]}` : "—"}</td>` +
      `<td class="n">${c ? betaMean(c.pooled).toFixed(2) : "—"}</td>` +
      `<td class="tiny">${flags.join(" ") || "—"}</td>` +
      `<td style="white-space:normal;max-width:60ch">${esc(cnd.summary)}</td>`;
    tr.style.cursor = "pointer";
    tr.onclick = () => { SEL.cand = cnd.name; openCandidate(cnd.name); renderTree(); };
    tb.append(tr);
  }
  t.append(tb);
  const sc = el("div", "scroll"); sc.append(t); reg.append(sc);
  v.append(reg);
}

// ---- the trace panel: what ran, on what hardware, under what conditions ----
function tracePanel(envsBy) {
  const card = el("div", "card");
  card.append(el("h2", null, "Trace — measurement environments"));
  card.append(el("p", "sub",
    "Every ledger row records the environment it ran under (`env`) and the timing method " +
    "(`timing.method`, samples, interleaving). These are the distinct environments in the " +
    "ledger. Click one to filter the tree to candidates measured under it; click again to clear."));
  const envs = SNAP.environments || [];
  if (!envs.length) {
    card.append(el("p", "empty-note", "this dashboard server predates the environments endpoint — restart it to get the trace"));
    return card;
  }
  const t = el("table", "grid");
  t.innerHTML = `<thead><tr><th class="n">env</th><th>device</th><th>arch</th><th>cuda</th>` +
    `<th>torch</th><th>triton</th><th>platform</th><th>clocks</th><th class="n">rows</th>` +
    `<th class="n">candidates</th><th>active</th></tr></thead>`;
  const tb = el("tbody");
  for (const e of envs) {
    const tr = el("tr");
    if (SEL.tree.envFilter === e.id) tr.className = "sel";
    tr.innerHTML = `<td class="n">#${e.id}</td>` +
      `<td>${esc(e.env.device ?? "—")}</td>` +
      `<td class="mono">${esc(e.env.cc ?? "—")}</td>` +
      `<td class="mono dim">${esc(e.env.cuda ?? "—")}</td>` +
      `<td class="mono dim">${esc(e.env.torch ?? "—")}</td>` +
      `<td class="mono dim">${esc(e.env.triton ?? "—")}</td>` +
      `<td class="dim">${esc(e.env.platform ?? "—")}</td>` +
      `<td>${e.env.clocks_locked === false ? '<span class="pill warn">unlocked</span>' : e.env.clocks_locked === true ? '<span class="pill ok">locked</span>' : "—"}</td>` +
      `<td class="n">${e.rows}</td><td class="n">${e.candidates}</td>` +
      `<td class="tiny dim">${ago(e.last_ts)} ago</td>`;
    tr.style.cursor = "pointer";
    tr.onclick = () => {
      SEL.tree.envFilter = SEL.tree.envFilter === e.id ? null : e.id;
      renderTree();
    };
    tb.append(tr);
  }
  t.append(tb);
  const sc = el("div", "scroll"); sc.append(t); card.append(sc);
  if (envs.some((e) => e.env.clocks_locked === false)) {
    card.append(el("p", "sub tiny",
      "clocks unlocked = WSL cannot lock GPU clocks (nvidia-smi -lgc fails), so every timing " +
      "here is minimum-of-N with candidate/baseline interleaved (docs/loop/method.md A2)."));
  }
  return card;
}

function treeLegend(top, frontier, fg) {
  const T = SEL.tree;
  const card = el("div", "card legend");

  const swatchRow = el("div", "legrow");
  if (T.colorBy === "cmp") {
    swatchRow.append(el("span", "k", "CMP posterior mean, Beta(1+W, 1+F) pooled"));
    const bar = svgEl("svg", { width: 220, height: 14, class: "rampbar" });
    for (let i = 0; i < 44; i++) bar.append(svgEl("rect", { x: i * 5, y: 0, width: 5, height: 14, fill: rgb(cmpColor(i / 43)) }));
    swatchRow.append(bar);
    swatchRow.append(el("span", "mono dim", "0.0 → 1.0"));
    swatchRow.append(el("span", "dim tiny", "higher = the sampler expects wins from this clade; hollow = no CMP stats (needs clean pad-0.0 rows)"));
  } else {
    swatchRow.append(el("span", "k", "geomean vs compiled — diverging at 1.00×"));
    const bar = svgEl("svg", { width: 264, height: 14, class: "rampbar" });
    // left third: the slower arm (0.5x -> 1x), rest: the faster arm (1x -> top), log both
    for (let i = 0; i < 12; i++) {
      const val = Math.exp(Math.log(0.5) * (1 - i / 11));
      bar.append(svgEl("rect", { x: i * 6, y: 0, width: 6, height: 14, fill: rgb(lineageColor(val, top)) }));
    }
    for (let i = 0; i < 32; i++) {
      const val = Math.exp((i / 31) * Math.log(Math.max(top, 1.05)));
      bar.append(svgEl("rect", { x: 72 + i * 6, y: 0, width: 6, height: 14, fill: rgb(lineageColor(val, top)) }));
    }
    swatchRow.append(bar);
    swatchRow.append(el("span", "mono dim", `0.50× · 1.00× · ${nf(top, 2)}×  (log)`));
    swatchRow.append(el("span", "dim tiny", "red arm = slower than the compiled baseline; blue arm = faster"));
  }
  card.append(swatchRow);

  const key = el("div", "legrow");
  const chip = (cls, label) => { const s = el("span", "legchip " + cls); const w = el("span", "legkey"); w.append(s); w.append(el("span", null, label)); return w; };
  key.append(chip("c-frontier", frontier
    ? `★ frontier — ${frontier} at ${fg ? nf(fg.geomean_compiled, 2) : "?"}× vs compiled`
    : "★ frontier"));
  key.append(chip("c-unmeasured", "hollow, dashed — no clean measurement yet"));
  key.append(chip("c-failed", "hatched — measured and NOTHING passed (a result, not an error)"));
  key.append(chip("c-partial", "n badge — n configs failed correctness on an otherwise-scoring run"));
  key.append(chip("c-leaf", "dot — dead end so far (nothing declares it as parent)"));
  key.append(chip("c-recomb", "dashed aqua — recombination (a second contributor, not the declared parent)"));
  key.append(chip("c-unsafe", "spine — known-unsafe: a wrong interpretation kept as lineage only"));
  key.append(chip("c-flag", "⚑ — created off the wrong branch point (recorded, not hidden)"));
  if (SEL.tree.envFilter != null) key.append(chip("c-offtrace", "faded — not measured under the selected environment"));
  card.append(key);
  return card;
}

// ------------------------------- drawer: one candidate, everything known about it
function betaSpark(entries) {
  // Beta(1+W, 1+F) posteriors, drawn as normalized pdfs. Log-space evaluation:
  // W,F reach the hundreds and x^(a-1) underflows long before that.
  const W = 280, H = 74, P = 8;
  const svg = svgEl("svg", { class: "beta", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  svg.append(svgEl("line", { class: "beta-axis", x1: P, y1: H - P, x2: W - P, y2: H - P }));
  for (const tx of [0, 0.5, 1]) {
    const x = P + tx * (W - 2 * P);
    svg.append(svgEl("line", { class: "beta-tick", x1: x, y1: H - P, x2: x, y2: H - P + 3 }));
    const t = svgEl("text", { class: "beta-lab", x, y: H - 1, "text-anchor": tx === 0 ? "start" : tx === 1 ? "end" : "middle" });
    t.textContent = String(tx);
    svg.append(t);
  }
  for (const { wf, cls, label } of entries) {
    const a = 1 + wf[0], b = 1 + wf[1];
    const N = 120, ls = [];
    let mx = -Infinity;
    for (let i = 0; i <= N; i++) {
      const x = (i + 0.5) / (N + 1);
      const l = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x);
      ls.push(l);
      if (l > mx) mx = l;
    }
    let d = "";
    let peakX = P, peakY = H - P;
    for (let i = 0; i <= N; i++) {
      const x = (i + 0.5) / (N + 1);
      const px = P + x * (W - 2 * P);
      const py = H - P - Math.exp(ls[i] - mx) * (H - 2 * P - 12);
      d += (i ? " L" : "M") + px.toFixed(1) + "," + py.toFixed(1);
      if (ls[i] === mx) { peakX = px; peakY = py; }
    }
    svg.append(svgEl("path", { class: "beta-curve " + cls, d }));
    const t = svgEl("text", {
      class: "beta-lab " + cls, x: Math.min(W - P, Math.max(P + 14, peakX)), y: Math.max(9, peakY - 3),
      "text-anchor": "middle",
    });
    t.textContent = `${label} μ=${(a / (a + b)).toFixed(2)}`;
    svg.append(t);
  }
  return svg;
}

function openCandidate(name) {
  const spec = (SNAP.lineage?.nodes || []).find((n) => n.name === name);
  const g = bestGroupFor(name);
  const anyG = anyGroupFor(name);
  const c = cmpOf(name);
  $("#drawer").hidden = false;
  $("#drawer-title").textContent = spec ? `${name} — generation ${spec.generation}` : name;
  const b = $("#drawer-body");
  b.innerHTML = "";

  // what it implements, first — this is what the compact node deliberately omits
  if (spec) { const p = el("p", "summary"); p.textContent = spec.summary; b.append(p); }

  const dl = el("dl", "kv");
  const add = (k, v2) => { dl.append(el("dt", null, k)); dl.append(el("dd", null, v2)); };
  add("declared parent", spec?.parent || "— (root of the lineage)");
  add("declared children", (spec?.children || []).join(", ") || "none yet (dead end)");
  if ((spec?.recombines || []).length) add("recombines", spec.recombines.join(", ") + "  (secondary contributor, not the declared parent)");
  if (spec?.known_unsafe) add("known-unsafe", "kept in the registry as lineage; not a submission candidate (tests/bench/test_lineage_invariants.py)");
  if (spec?.topology_violation) add("git topology", "branched off the wrong commit after the discipline existed — recorded, not hidden (tests/bench/test_lineage_topology.py)");
  if (g) {
    add("configs", `${g.configs_passed} passed / ${g.configs_measured} measured`);
    add("geomean", `${sx(g.geomean_compiled)} vs compiled · ${sx(g.geomean)} vs eager`);
    add("weighted score", `${nf(g.weighted_score_compiled, 3)} compiled · ${nf(g.weighted_score, 3)} eager`);
  } else if (anyG && failsOf(anyG)) {
    add("measurement", `measured — nothing passed (${anyG.configs_measured} config(s), every one failed)`);
  } else if (anyG) {
    add("measurement", "probe rows only — no pass/fail verdict available at these shapes (config 14 protocol)");
  } else {
    add("measurement", "none recorded from a clean tree");
  }
  b.append(dl);

  // CMP: the numbers Thompson sampling draws from
  b.append(el("h2", null, "CMP — what the parent sampler sees"));
  if (c) {
    const dl2 = el("dl", "kv");
    const add2 = (k, v2) => { dl2.append(el("dt", null, k)); dl2.append(el("dd", null, v2)); };
    add2("own", `${c.own[0]} wins / ${c.own[1]} failures (this candidate's rows only)`);
    add2("pooled", `${c.pooled[0]} wins / ${c.pooled[1]} failures (its whole declared subtree)`);
    add2("sampler draw", `Beta(1+${c.pooled[0]}, 1+${c.pooled[1]}) → posterior mean ${betaMean(c.pooled).toFixed(3)}`);
    b.append(dl2);
    b.append(betaSpark([
      { wf: c.pooled, cls: "b-pooled", label: "pooled" },
      { wf: c.own, cls: "b-own", label: "own" },
    ]));
    if (SNAP.cmp?.criterion) b.append(el("p", "sub tiny", SNAP.cmp.criterion));
  } else {
    b.append(el("p", "sub", SNAP.cmp && !SNAP.cmp.error
      ? "no CMP stats: CMP counts only clean, passing pad-0.0 rows, and this candidate has none."
      : "CMP stats unavailable" + (SNAP.cmp?.error ? ` — ${SNAP.cmp.error}` : " — this dashboard server predates the CMP endpoint.")));
  }

  // Trace: what ran, where, and how it was timed
  const trace = g || anyG;
  if (trace) {
    b.append(el("h2", null, "Trace — how this was measured"));
    const dl3 = el("dl", "kv");
    const add3 = (k, v2) => { dl3.append(el("dt", null, k)); dl3.append(el("dd", null, v2)); };
    add3("commit", `${trace.short_sha} on ${trace.branch || "—"}`);
    add3("padding", trace.padding_ratio.toFixed(1));
    const envs = (trace.env_ids || []).map((id) => (SNAP.environments || []).find((e) => e.id === id)).filter(Boolean);
    for (const e of envs) {
      add3(`env #${e.id}`, `${e.env.device ?? "?"} (${e.env.cc ?? "?"}) · cuda ${e.env.cuda ?? "?"} · torch ${e.env.torch ?? "?"}` +
        (e.env.triton ? ` · triton ${e.env.triton}` : "") +
        (e.env.platform ? ` · ${e.env.platform}` : "") +
        (e.env.clocks_locked === false ? " · clocks UNLOCKED" : ""));
    }
    if (!envs.length && trace.env_ids?.length === 0) add3("env", "no env recorded on these rows");
    const pcs = Object.values(trace.per_config || {});
    const methods = [...new Set(pcs.map((r) => r.method).filter(Boolean))];
    if (methods.length) {
      const sample = pcs.find((r) => r.method);
      add3("timing", methods.join(", ") +
        (sample?.samples ? ` · n=${sample.samples}` : "") +
        (sample?.interleaved ? " · interleaved" : "") +
        (sample?.arms_isolated ? " · arms isolated" : ""));
    }
    const conds = [...new Set(pcs.map((r) => r.conditions).filter(Boolean))];
    if (conds.length) add3("conditions", conds[0] + (conds.length > 1 ? `  (+${conds.length - 1} more, see per-config)` : ""));
    add3("last measured", `${trace.latest_ts} (${ago(trace.latest_ts)} ago)`);
    b.append(dl3);
  }

  if (g) {
    b.append(el("h2", null, "Per config"));
    const sc = el("div", "scroll"); sc.append(perConfigTable(g)); b.append(sc);
  } else if (anyG) {
    b.append(el("h2", null, failsOf(anyG) ? "Per config (all failing)" : "Per config (probe rows)"));
    const sc = el("div", "scroll"); sc.append(perConfigTable(anyG)); b.append(sc);
  }
}

// -------------------------------------------------------------- scoreboard
const SB_COLS = [
  ["candidate", "candidate", "s"],
  ["padding_ratio", "pad", "n"],
  ["short_sha", "commit", "s"],
  ["branch", "branch", "s"],
  ["configs_measured", "cfgs", "n"],
  ["configs_passed", "pass", "n"],
  ["rows", "rows", "n"],
  ["geomean", "geomean vs eager", "n"],
  ["geomean_compiled", "geomean vs compiled", "n"],
  ["weighted_score", "score vs eager", "n"],
  ["weighted_score_compiled", "score vs compiled", "n"],
  ["latest_ts", "last measured", "s"],
];
function renderScoreboard() {
  const v = $("#view-scoreboard");
  v.innerHTML = "";
  v.append(el("h2", null, "Scoreboard — one row per (commit, candidate, padding_ratio)"));
  v.append(el("p", "sub",
    "Padding is a measurement CONDITION, not a detail: pad 0.0 and pad 0.5 are never pooled. Weighted score clips each config at 3× and scores an unmeasured config 1.0 over all 14 rows of the matrix, so not measuring is never rewarded. Rows recorded from a dirty tree are excluded. Click a row for per-config detail."));

  const card = el("div", "card scroll");
  const t = el("table", "grid");
  const thead = el("thead"); const htr = el("tr");
  for (const [key, label, kind] of SB_COLS) {
    const th = el("th", kind === "n" ? "n" : "", label);
    th.style.cursor = "pointer";
    th.onclick = () => {
      SEL.sbSort = { key, dir: SEL.sbSort.key === key ? -SEL.sbSort.dir : (kind === "n" ? -1 : 1) };
      renderScoreboard();
    };
    if (SEL.sbSort.key === key) th.textContent = label + (SEL.sbSort.dir < 0 ? " ▾" : " ▴");
    htr.append(th);
  }
  thead.append(htr); t.append(thead);

  const rows = [...SNAP.scoreboard].sort((a, b) => {
    const k = SEL.sbSort.key, d = SEL.sbSort.dir;
    const x = a[k], y = b[k];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * d;
  });
  const bestGeo = Math.max(...rows.map((r) => r.geomean_compiled ?? 0));
  const tb = el("tbody");
  for (const g of rows) {
    const tr = el("tr");
    if (SEL.node === g.commit_sha) tr.className = "sel";
    const fail = g.configs_measured - g.configs_passed;
    tr.innerHTML =
      `<td class="mono">${esc(g.candidate || "—")}</td>` +
      `<td class="n">${g.padding_ratio.toFixed(1)}</td>` +
      `<td class="mono dim">${esc(g.short_sha)}</td>` +
      `<td class="mono dim tiny">${esc(g.branch || "—")}</td>` +
      `<td class="n">${g.configs_measured}</td>` +
      `<td class="n">${fail ? `<span class="pill bad">${g.configs_passed}/${g.configs_measured}</span>` : `<span class="pill ok">${g.configs_passed}</span>`}</td>` +
      `<td class="n dim">${g.rows}${g.sweep ? ' <span class="pill violet" title="more rows than configs: a parameter sweep, last write per config wins">sweep</span>' : ""}</td>` +
      `<td class="n" style="color:var(--faint)">${sx(g.geomean)}</td>` +
      `<td class="n hon ${g.geomean_compiled === bestGeo && bestGeo > 0 ? "best" : ""}">${sx(g.geomean_compiled)}</td>` +
      `<td class="n" style="color:var(--faint)">${nf(g.weighted_score, 3)}</td>` +
      `<td class="n">${nf(g.weighted_score_compiled, 3)}</td>` +
      `<td class="mono dim tiny">${ago(g.latest_ts)} ago</td>`;
    tr.style.cursor = "pointer";
    tr.onclick = () => openGroup(g);
    tb.append(tr);
  }
  t.append(tb); card.append(t); v.append(card);

  const note = el("div", "card");
  note.append(el("h2", null, "Reading this table"));
  const ul = el("ul", "md");
  ul.innerHTML =
    `<li><b>geomean vs eager</b> is what the ledger records: <code>timing.speedup</code>, computed against the unmodified eager baseline.</li>` +
    `<li><b>geomean vs compiled</b> is recomputed here as <code>baseline_compiled.candidate_ms / candidate_ms</code> per config — the honest number against torch.compile(max-autotune).</li>` +
    `<li>A candidate measured at two padding ratios appears twice, deliberately.</li>` +
    `<li><b>cfgs</b> counts DISTINCT config ids; <b>rows</b> is the raw ledger row count. A group tagged <span class="pill violet">sweep</span> wrote several rows for the same config (a parameter search) — the last row per config wins, as in <code>bench/ledger.py</code>.</li>`;
  note.append(ul);
  v.append(note);
}

// ----------------------------------------------------------------- heatmap
function renderHeatmap() {
  const v = $("#view-heatmap");
  v.innerHTML = "";
  v.append(el("h2", null, "Per-config speedup — candidates × the 13 runnable configs"));
  v.append(el("p", "sub",
    "Colour diverges at 1.00× on a LOG scale: blue = faster than the baseline, red = slower; linear would flatten the whole grid into one shade. Config 14 is excluded: it is a feasibility probe, not a timed config. Each cell carries BOTH numbers — the large one is the selected baseline, the small one the other. Unmeasured cells are outlined and empty; a failed cell is hatched."));

  const controls = el("div", "controls");
  const seg = el("div", "seg");
  for (const [k, label] of [["compiled", "large = vs compiled baseline"], ["eager", "large = vs eager baseline"]]) {
    const b = el("button", "btn", label);
    b.setAttribute("aria-pressed", String(SEL.heatMetric === k));
    b.onclick = () => { SEL.heatMetric = k; renderHeatmap(); };
    seg.append(b);
  }
  controls.append(seg);
  v.append(controls);

  const card = el("div", "card scroll");
  const t = el("table", "heat");
  const cfgById = new Map(SNAP.matrix.map((c) => [c.id, c]));
  const thead = el("thead"); const htr = el("tr");
  htr.append(el("th", "rowh", "candidate"));
  for (const cid of SNAP.heatmap.configs) {
    const c = cfgById.get(cid);
    const th = el("th", "cfgh");
    th.innerHTML = `#${cid}<span class="r">${esc((c?.regime || "").replace(/_/g, " "))}</span>`;
    th.title = c ? `config ${cid} · ${c.regime}\nB=${c.batch_size} D=${c.d_model} H=${c.heads} hd=${c.head_dim} S=${c.seq_len} L=${c.layers}` : "";
    htr.append(th);
  }
  htr.append(el("th", "cfgh", "geomean"));
  thead.append(htr); t.append(thead);

  const tb = el("tbody");
  for (const lane of SNAP.heatmap.lanes) {
    const tr = el("tr");
    const rh = el("th", "rowh");
    rh.innerHTML = `${esc(lane.candidate)} <span class="pad">pad ${lane.padding_ratio.toFixed(1)}</span> <span class="pad">${esc(lane.short_sha)}</span>`;
    tr.append(rh);
    for (const cid of SNAP.heatmap.configs) {
      const cell = lane.cells[cid];
      const td = el("td", "cell");
      if (!cell) { td.classList.add("empty"); td.textContent = "·"; td.title = `config ${cid}: not measured for ${lane.candidate} @ pad ${lane.padding_ratio}`; tr.append(td); continue; }
      if (!cell.passed) {
        td.classList.add("fail", "clickable");
        td.innerHTML = `<span class="a">${esc(cell.status || "fail")}</span>`;
        td.title = `config ${cid} · ${cell.status}` + (cell.max_abs != null ? `\nmax_abs ${sci(cell.max_abs)} vs budget 2.00e-3` : "");
        td.onclick = () => openCell(lane, cell);
        tr.append(td); continue;
      }
      const big = SEL.heatMetric === "compiled" ? cell.speedup_compiled : cell.speedup;
      const small = SEL.heatMetric === "compiled" ? cell.speedup : cell.speedup_compiled;
      const c = heatColor(Number.isFinite(big) ? big : 1);
      td.style.background = rgb(c);
      td.style.color = readable(c);
      td.classList.add("clickable");
      td.innerHTML = `<span class="a">${sx(big)}</span><span class="b">${sx(small)}</span>`;
      td.title = `config ${cid} · ${lane.candidate} @ pad ${lane.padding_ratio}\n` +
        `vs eager    ${sx(cell.speedup)}   (${ms(cell.baseline_ms)} → ${ms(cell.candidate_ms)} ms)\n` +
        `vs compiled ${sx(cell.speedup_compiled)}   (${ms(cell.compiled_baseline_ms)} → ${ms(cell.candidate_ms)} ms)`;
      td.onclick = () => openCell(lane, cell);
      tr.append(td);
    }
    const gt = el("td", "cell");
    const gbig = SEL.heatMetric === "compiled" ? lane.geomean_compiled : lane.geomean;
    const gsmall = SEL.heatMetric === "compiled" ? lane.geomean : lane.geomean_compiled;
    const gc = heatColor(Number.isFinite(gbig) ? gbig : 1);
    gt.style.background = rgb(gc); gt.style.color = readable(gc);
    gt.style.outline = "1px solid var(--line)";
    gt.innerHTML = `<span class="a">${sx(gbig)}</span><span class="b">${sx(gsmall)}</span>`;
    tr.append(gt);
    tb.append(tr);
  }
  t.append(tb); card.append(t);

  const scale = el("div", "scale");
  scale.append(el("span", "lab", "0.5×"));
  const bar = el("div", "bar");
  for (let i = 0; i < 6; i++) {
    const sp = Math.exp(Math.log(0.5) * (1 - i / 5));
    const b = el("i"); b.style.background = rgb(heatColor(sp)); b.style.width = "10px"; bar.append(b);
  }
  for (let i = 0; i <= 24; i++) {
    const sp = Math.exp((i / 24) * Math.log(TOP));
    const b = el("i"); b.style.background = rgb(heatColor(sp)); b.style.width = "10px"; bar.append(b);
  }
  scale.append(bar);
  scale.append(el("span", "lab", "1× at the seam · 34×  (log)"));
  card.append(scale);

  const leg = el("div", "legend");
  leg.innerHTML =
    `<span><span class="sw empty"></span>not measured</span>` +
    `<span><span class="sw fail"></span>measured and failed (incorrect / oom / crash)</span>` +
    `<span><span class="sw" style="background:${rgb(heatColor(1))};border:1px solid var(--line)"></span>1.00× — parity with the baseline</span>` +
    `<span><span class="sw" style="background:${rgb(heatColor(0.6))}"></span>slower than the baseline</span>`;
  card.append(leg);
  v.append(card);
}

// --------------------------------------------------------------- baselines
function renderBaselines() {
  const v = $("#view-baselines");
  v.innerHTML = "";
  v.append(el("h2", null, "Two baselines, side by side"));
  v.append(el("p", "sub",
    "The ledger's speedups are computed against EAGER. CLAUDE.md rule 5 makes torch.compile(max-autotune) the baseline a submission must beat, so the compiled column is the honest one. The gap is large and is never collapsed into a single headline here."));

  const grid = el("div", "row2");

  const c1 = el("div", "card scroll");
  c1.append(el("h2", null, "Per config — what each baseline costs"));
  const t1 = el("table", "grid");
  t1.innerHTML = `<thead><tr><th class="n">cfg</th><th>regime</th><th class="n">eager ms</th><th class="n">compiled ms</th>` +
    `<th class="n">compile speedup</th></tr></thead>`;
  const b1 = el("tbody");
  const cfgById = new Map(SNAP.matrix.map((c) => [c.id, c]));
  for (const c of SNAP.matrix) {
    const e = SNAP.baselines.eager_ms[c.id], k = SNAP.baselines.compiled_ms[c.id];
    const tr = el("tr");
    tr.innerHTML = `<td class="n">${c.id}</td><td class="dim tiny">${esc(c.regime.replace(/_/g, " "))}</td>` +
      `<td class="n">${ms(e)}</td><td class="n">${ms(k)}</td>` +
      `<td class="n" style="color:var(--warn)">${e && k ? sx(e / k) : "—"}</td>`;
    b1.append(tr);
  }
  t1.append(b1); c1.append(t1);
  c1.append(el("p", "sub tiny", `eager rows: ${SNAP.baselines.eager_rows} · compiled rows: ${SNAP.baselines.compiled_rows}. A missing compiled entry means every "vs compiled" number for that config is unavailable, not 1×.`));
  grid.append(c1);

  const c2 = el("div", "card scroll");
  c2.append(el("h2", null, "Per candidate — the same work, two denominators"));
  const t2 = el("table", "grid");
  t2.innerHTML = `<thead><tr><th>candidate</th><th class="n">pad</th><th class="n">geomean vs eager</th>` +
    `<th class="n">geomean vs compiled</th><th class="n">ratio</th></tr></thead>`;
  const b2 = el("tbody");
  for (const lane of SNAP.heatmap.lanes) {
    const ratio = lane.geomean && lane.geomean_compiled ? lane.geomean / lane.geomean_compiled : null;
    const tr = el("tr");
    tr.innerHTML = `<td class="mono">${esc(lane.candidate)}</td><td class="n">${lane.padding_ratio.toFixed(1)}</td>` +
      `<td class="n" style="color:var(--faint)">${sx(lane.geomean)}</td>` +
      `<td class="n hon"><b>${sx(lane.geomean_compiled)}</b></td>` +
      `<td class="n" style="color:var(--warn)">${ratio ? nf(ratio, 2) + "× flattered" : "—"}</td>`;
    b2.append(tr);
  }
  t2.append(b2); c2.append(t2);
  grid.append(c2);
  v.append(grid);

  const c3 = el("div", "card scroll");
  c3.append(el("h2", null, "Best measured candidate per config, against both"));
  const t3 = el("table", "grid");
  t3.innerHTML = `<thead><tr><th class="n">cfg</th><th>best candidate</th><th class="n">pad</th><th class="n">ms</th>` +
    `<th class="n">vs eager</th><th class="n">vs compiled</th></tr></thead>`;
  const b3 = el("tbody");
  for (const cid of SNAP.heatmap.configs) {
    let best = null;
    for (const lane of SNAP.heatmap.lanes) {
      const cell = lane.cells[cid];
      if (!cell || !cell.passed || !Number.isFinite(cell.candidate_ms)) continue;
      if (!best || cell.candidate_ms < best.cell.candidate_ms) best = { lane, cell };
    }
    const tr = el("tr");
    if (!best) { tr.innerHTML = `<td class="n">${cid}</td><td class="dim" colspan="5">no passing measurement</td>`; }
    else {
      tr.innerHTML = `<td class="n">${cid}</td><td class="mono">${esc(best.lane.candidate)}</td>` +
        `<td class="n">${best.lane.padding_ratio.toFixed(1)}</td><td class="n">${ms(best.cell.candidate_ms)}</td>` +
        `<td class="n" style="color:var(--faint)">${sx(best.cell.speedup)}</td>` +
        `<td class="n hon">${sx(best.cell.speedup_compiled)}</td>`;
    }
    b3.append(tr);
  }
  t3.append(b3); c3.append(t3);
  v.append(c3);
}

// ----------------------------------------------------------------- failures
function renderFailures() {
  const v = $("#view-failures");
  v.innerHTML = "";
  v.append(el("h2", null, "Failures — these are results, not errors"));
  v.append(el("p", "sub",
    "Every row whose status is not ok, or whose correctness did not pass. max_abs is shown against the locked 2.00e-3 budget; a bar at or above 1.0 means the row spent the whole budget. Rows recorded from a dirty tree are marked."));
  if (!SNAP.failures.length) { v.append(el("p", "empty-note", "no failing rows in the ledger")); return; }

  const card = el("div", "card scroll");
  const t = el("table", "grid");
  t.innerHTML = `<thead><tr><th>when</th><th>candidate</th><th class="n">cfg</th><th class="n">pad</th>` +
    `<th>status</th><th class="n">max_abs</th><th class="n">/ budget</th><th class="n">max_rel</th>` +
    `<th class="n">failed elems</th><th>commit</th><th>note</th></tr></thead>`;
  const tb = el("tbody");
  for (const f of SNAP.failures) {
    const frac = f.max_abs != null ? f.max_abs / f.budget : null;
    const cls = f.status === "incorrect" ? "bad" : f.status === "oom" ? "warn" : "bad";
    const tr = el("tr");
    tr.innerHTML =
      `<td class="mono dim tiny">${ago(f.ts)} ago</td>` +
      `<td class="mono">${esc(f.candidate || "—")}${f.dirty ? ' <span class="pill warn">dirty</span>' : ""}</td>` +
      `<td class="n">${f.config_id}</td><td class="n">${f.padding_ratio.toFixed(1)}</td>` +
      `<td><span class="pill ${cls}">${esc(f.status)}</span></td>` +
      `<td class="n">${sci(f.max_abs)}</td>` +
      `<td class="n" style="color:${frac == null ? "var(--faint)" : frac >= 1 ? "var(--crit)" : "var(--warn)"}">${frac == null ? "—" : nf(frac, 2)}</td>` +
      `<td class="n dim">${sci(f.max_rel)}</td>` +
      `<td class="n dim">${f.failed_elements ?? "—"}</td>` +
      `<td class="mono dim tiny">${esc(f.commit_short)}</td>` +
      `<td class="dim tiny" style="white-space:normal;max-width:52ch">${esc((f.notes || "").split("\n")[0].slice(0, 160))}</td>`;
    tb.append(tr);
  }
  t.append(tb); card.append(t); v.append(card);
}

// ---------------------------------------------------------------- learnings
async function renderLearnings() {
  const v = $("#view-learnings");
  v.innerHTML = "";
  v.append(el("h2", null, "Findings — the loop's long-term memory"));
  v.append(el("p", "sub", "docs/findings/ re-read on every poll. 00-learnings.md is what the loop reads at the start of each iteration."));

  const list = el("div", "doclist");
  for (const d of SNAP.findings.docs) {
    const b = el("button", "btn", d.file);
    b.setAttribute("aria-pressed", String(SEL.doc === d.file));
    b.title = d.title;
    b.onclick = async () => { SEL.doc = d.file; await renderLearnings(); };
    list.append(b);
  }
  v.append(list);

  const card = el("div", "card");
  const body = el("div", "md");
  let text = null;
  if (SEL.doc === "00-learnings.md" && SNAP.findings.learnings) text = SNAP.findings.learnings;
  else if (docCache.has(SEL.doc)) text = docCache.get(SEL.doc);
  if (text == null) {
    body.textContent = "loading…";
    try {
      const r = await fetch("/api/finding?file=" + encodeURIComponent(SEL.doc));
      const j = await r.json();
      text = j.text || "(empty)";
      docCache.set(SEL.doc, text);
    } catch (e) { text = "failed to load: " + e.message; }
  }
  body.innerHTML = md(text);
  card.append(body); v.append(card);
}

// ------------------------------------------------------------------- drawer
function closeDrawer() { $("#drawer").hidden = true; }
$("#drawer-close").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

function perConfigTable(g) {
  const cfgById = new Map(SNAP.matrix.map((c) => [c.id, c]));
  const t = el("table", "grid");
  t.innerHTML = `<thead><tr><th class="n">cfg</th><th>regime</th><th>status</th><th class="n">cand ms</th>` +
    `<th class="n">eager ms</th><th class="n">compiled ms</th><th class="n">vs eager</th><th class="n">vs compiled</th>` +
    `<th class="n">max_abs</th><th class="n">peak MB</th><th class="tiny">method</th></tr></thead>`;
  const tb = el("tbody");
  for (const c of SNAP.matrix) {
    const r = g.per_config[c.id];
    const tr = el("tr");
    if (!r) { tr.innerHTML = `<td class="n">${c.id}</td><td class="dim tiny">${esc(c.regime.replace(/_/g, " "))}</td><td class="dim" colspan="9">not measured</td>`; tb.append(tr); continue; }
    const vd = r.verdict ?? (r.passed ? "pass" : "fail");
    tr.innerHTML =
      `<td class="n">${c.id}</td><td class="dim tiny">${esc(c.regime.replace(/_/g, " "))}</td>` +
      `<td><span class="pill ${vd === "pass" ? "ok" : vd === "none" ? "mute" : "bad"}">${esc(r.status)}</span></td>` +
      `<td class="n">${ms(r.candidate_ms)}</td><td class="n dim">${ms(r.baseline_ms ?? SNAP.baselines.eager_ms[c.id])}</td>` +
      `<td class="n dim">${ms(r.compiled_baseline_ms)}</td>` +
      `<td class="n" style="color:var(--faint)">${sx(r.speedup)}</td>` +
      `<td class="n hon">${sx(r.speedup_compiled)}</td>` +
      `<td class="n">${sci(r.max_abs)}</td><td class="n dim">${r.peak_MB != null ? nf(r.peak_MB, 0) : "—"}</td>` +
      `<td class="tiny dim">${esc(r.method || "—")}${r.samples ? " n=" + r.samples : ""}${r.interleaved ? " interleaved" : ""}${r.arms_isolated ? " isolated" : ""}</td>`;
    tb.append(tr);
  }
  t.append(tb);
  return t;
}

function openGroup(g) {
  SEL.node = g.commit_sha;
  $("#drawer").hidden = false;
  $("#drawer-title").textContent = `${g.candidate} @ pad ${g.padding_ratio.toFixed(1)} · ${g.short_sha}`;
  const b = $("#drawer-body");
  b.innerHTML = "";
  const spec = SNAP.candidates.find((c) => c.name === g.candidate);
  if (spec) { const p = el("p", "summary"); p.textContent = spec.summary; b.append(p); }
  const dl = el("dl", "kv");
  const add = (k, v) => { dl.append(el("dt", null, k)); dl.append(el("dd", null, v)); };
  add("commit", g.commit_sha);
  add("branch", g.branch || "—");
  add("generation", spec ? String(spec.generation) : "—");
  add("parent", spec?.parent || "—");
  add("measured / passed", `${g.configs_measured} / ${g.configs_passed}`);
  add("geomean vs eager", sx(g.geomean));
  add("geomean vs compiled", sx(g.geomean_compiled));
  add("weighted score", `${nf(g.weighted_score, 3)} eager · ${nf(g.weighted_score_compiled, 3)} compiled`);
  const envs = (g.env_ids || []).map((id) => (SNAP.environments || []).find((e) => e.id === id)).filter(Boolean);
  for (const e of envs) {
    add(`env #${e.id}`, `${e.env.device ?? "?"} (${e.env.cc ?? "?"}) · cuda ${e.env.cuda ?? "?"} · torch ${e.env.torch ?? "?"}` +
      (e.env.clocks_locked === false ? " · clocks UNLOCKED" : ""));
  }
  add("last measured", `${g.latest_ts} (${ago(g.latest_ts)} ago)`);
  b.append(dl);
  b.append(el("h2", null, "Per config"));
  b.append(el("div", "scroll", "").appendChild(perConfigTable(g)).parentNode);
}

function openCell(lane, cell) {
  const g = SNAP.scoreboard.find((x) => x.commit_sha === lane.commit_sha && x.candidate === lane.candidate && x.padding_ratio === lane.padding_ratio);
  if (g) openGroup(g);
}

// -------------------------------------------------------------------- render
function render() {
  renderStatus();
  renderTabs();
  if (!SNAP) return;
  if (SEL.tab === "tree") renderTree();
  else if (SEL.tab === "scoreboard") renderScoreboard();
  else if (SEL.tab === "heatmap") renderHeatmap();
  else if (SEL.tab === "baselines") renderBaselines();
  else if (SEL.tab === "failures") renderFailures();
  else if (SEL.tab === "learnings") renderLearnings();
}

// ----------------------------------------------------------------------- SSE
let STREAM_OK = false;
function connect() {
  const es = new EventSource("/api/stream");
  es.addEventListener("open", () => { STREAM_OK = true; renderStatus(); });
  es.addEventListener("snapshot", (ev) => {
    try { SNAP = JSON.parse(ev.data); STREAM_OK = true; render(); }
    catch (e) { console.error("bad snapshot", e); }
  });
  es.addEventListener("error", () => { STREAM_OK = false; renderStatus(); });
  return es;
}

try {
  const saved = localStorage.getItem("ratchet-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch {}
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => render());

fetch("/api/snapshot").then((r) => r.json()).then((s) => { SNAP = s; render(); }).catch(() => {});
connect();

// ---------------------------------------------------------------- test seam
// No browser is available in this environment, so `dashboard/test/render.test.mjs`
// drives THIS file against a real snapshot in a minimal DOM. Exports only -- the
// browser path above is unchanged by their presence.
export function __setSnapshot(s) { SNAP = s; }
export { render, renderTree, SEL };

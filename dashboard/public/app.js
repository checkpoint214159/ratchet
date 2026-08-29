// ratchet dashboard — client. Vanilla ES modules, no build step, no dependencies.

const $  = (s, r = document) => r.querySelector(s);
const el = (t, cls, txt) => { const n = document.createElement(t); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let SNAP = null;
let SEL = { tab: "tree", node: null, heatMetric: "compiled", sbSort: { key: "weighted_score", dir: -1 }, doc: "00-learnings.md" };
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

// ------------------------------------------------------- log-scale heat ramp
// Speedups span ~1x to ~34x. Linear flattens everything into one shade, so the
// ramp is in log space with 34x pinned to the top stop.
const TOP = 34;
const RAMP_LIGHT = [[0, [240,243,245]], [.25, [189,226,217]], [.50, [87,187,167]], [.72, [4,140,116]], [.88, [61,110,146]], [1, [107,92,165]]];
const RAMP_DARK  = [[0, [32,43,53]],   [.25, [20,84,74]],    [.50, [6,150,128]],  [.72, [79,184,166]], [.88, [129,114,190]], [1, [176,162,224]]];
const SLOW_LIGHT = [[0, [240,243,245]], [.5, [233,169,155]], [1, [179,64,44]]];
const SLOW_DARK  = [[0, [32,43,53]],    [.5, [138,60,47]],   [1, [207,91,69]]];
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
  if (sp >= 1) return rampAt(dark ? RAMP_DARK : RAMP_LIGHT, Math.log(sp) / Math.log(TOP));
  return rampAt(dark ? SLOW_DARK : SLOW_LIGHT, Math.min(1, Math.log(1 / sp) / Math.log(4)));
}
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const readable = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 150 ? "#151C24" : "#FFFFFF");

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
    if (id === "tree" && SNAP) b.append(el("span", "badge", String(SNAP.tree.nodes.length)));
    if (id === "scoreboard" && SNAP) b.append(el("span", "badge", String(SNAP.scoreboard.length)));
    b.onclick = () => { SEL.tab = id; render(); };
    nav.append(b);
  }
  for (const [id] of TABS) { const v = $("#view-" + id); if (v) v.hidden = SEL.tab !== id; }
}

// ------------------------------------------------------------- evolution tree
function groupsAt(sha) { return SNAP.scoreboard.filter((g) => g.commit_sha === sha); }
function primaryAt(sha) {
  const gs = groupsAt(sha).filter((g) => g.candidate && g.candidate !== "baseline");
  if (!gs.length) return null;
  return gs.reduce((a, b) => (b.weighted_score > a.weighted_score ? b : a));
}

function renderTree() {
  const v = $("#view-tree");
  v.innerHTML = "";
  const head = el("div");
  head.append(el("h2", null, "Evolution tree — git ancestry is the lineage"));
  head.append(el("p", "sub",
    "Nodes are commits that carry measurements, merges, or branch tips; edges are nearest-interesting ancestry over the real DAG, so the merge keeps both of its parents. Amber = merge. Click a node for its per-config detail."));
  v.append(head);

  const nodes = SNAP.tree.nodes.map((n) => ({ ...n }));
  const edges = SNAP.tree.edges;
  if (!nodes.length) { v.append(el("p", "empty-note", "no commits to draw")); return; }

  const byS = new Map(nodes.map((n) => [n.sha, n]));
  // lane assignment: a child inherits its first parent's lane when free
  const parents = new Map(nodes.map((n) => [n.sha, []]));
  for (const e of edges) parents.get(e.to)?.push(e.from);
  const maxDepth = Math.max(...nodes.map((n) => n.depth));
  const occupied = new Map(); // depth -> Set(lane)
  const take = (d, want) => {
    const set = occupied.get(d) || new Set();
    let lane = want;
    while (set.has(lane)) lane++;
    set.add(lane); occupied.set(d, set);
    return lane;
  };
  for (const n of nodes.sort((a, b) => a.depth - b.depth || (a.when < b.when ? -1 : 1))) {
    const ps = (parents.get(n.sha) || []).map((s) => byS.get(s)).filter(Boolean);
    const want = ps.length ? Math.min(...ps.map((p) => p.lane ?? 0)) : 0;
    n.lane = take(n.depth, want);
  }
  const maxLane = Math.max(...nodes.map((n) => n.lane));

  const W = 250, H = 54, GX = 34, GY = 30, PAD = 18;
  const width = PAD * 2 + (maxLane + 1) * (W + GX);
  const height = PAD * 2 + (maxDepth + 1) * (H + GY);
  const cx = (n) => PAD + n.lane * (W + GX);
  const cy = (n) => PAD + n.depth * (H + GY);

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.id = "tree";
  svg.setAttribute("width", width); svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  for (const e of edges) {
    const a = byS.get(e.from), b = byS.get(e.to);
    if (!a || !b) continue;
    const x1 = cx(a) + W / 2, y1 = cy(a) + H, x2 = cx(b) + W / 2, y2 = cy(b);
    const p = document.createElementNS(NS, "path");
    const mid = (y1 + y2) / 2;
    p.setAttribute("d", `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`);
    p.setAttribute("class", "edge" + (b.isMerge ? " merge" : ""));
    svg.append(p);
  }

  for (const n of nodes) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "node" + (n.rows ? " measured" : "") + (n.isMerge ? " mergenode" : "") + (SEL.node === n.sha ? " sel" : ""));
    g.setAttribute("transform", `translate(${cx(n)},${cy(n)})`);
    const r = document.createElementNS(NS, "rect");
    r.setAttribute("class", "box"); r.setAttribute("width", W); r.setAttribute("height", H); r.setAttribute("rx", 3);
    g.append(r);

    const prim = primaryAt(n.sha);
    const label = prim ? prim.candidate : (n.candidates[0] || (n.isMerge ? "merge" : "—"));
    const t1 = document.createElementNS(NS, "text");
    t1.setAttribute("class", "nlabel"); t1.setAttribute("x", 9); t1.setAttribute("y", 17);
    t1.textContent = label.length > 24 ? label.slice(0, 23) + "…" : label;
    g.append(t1);

    const t2 = document.createElementNS(NS, "text");
    t2.setAttribute("class", "nsub"); t2.setAttribute("x", 9); t2.setAttribute("y", 31);
    t2.textContent = `${n.short}  ${n.rows ? n.rows + " rows" : "no rows"}`;
    g.append(t2);

    const t3 = document.createElementNS(NS, "text");
    t3.setAttribute("class", "nnum"); t3.setAttribute("x", 9); t3.setAttribute("y", 45);
    if (prim) {
      t3.textContent = `geo ${sx(prim.geomean)} eager / ${sx(prim.geomean_compiled)} compiled`;
      t3.setAttribute("fill", "var(--teal)");
    } else { t3.textContent = (n.subject || "").slice(0, 34); t3.setAttribute("fill", "var(--faint)"); }
    g.append(t3);

    if (n.tips && n.tips.length) {
      const t4 = document.createElementNS(NS, "text");
      t4.setAttribute("class", "tip"); t4.setAttribute("x", W - 9); t4.setAttribute("y", 17);
      t4.setAttribute("text-anchor", "end");
      t4.textContent = n.tips.join(" ");
      g.append(t4);
    }
    const title = document.createElementNS(NS, "title");
    title.textContent = `${n.short}  ${n.subject}\n${n.when || ""}${n.isMerge ? "\n(merge commit — two parents)" : ""}`;
    g.append(title);
    g.style.cursor = "pointer";
    g.onclick = () => { SEL.node = n.sha; openCommit(n.sha); renderTree(); };
    svg.append(g);
  }

  const wrap = el("div", "card treewrap");
  wrap.append(svg);
  v.append(wrap);

  const reg = el("div", "card");
  reg.append(el("h2", null, "Candidate registry (bench/candidates REGISTRY)"));
  reg.append(el("p", "sub", "Declared generation and parent. Lineage for g8+ comes from git; the first entries predate the branch protocol and state their parent here."));
  const t = el("table", "grid");
  t.innerHTML = `<thead><tr><th class="n">gen</th><th>candidate</th><th>parent</th><th>summary</th></tr></thead>`;
  const tb = el("tbody");
  for (const c of SNAP.candidates) {
    const tr = el("tr");
    tr.innerHTML = `<td class="n">${c.generation}</td><td class="mono">${esc(c.name)}</td>` +
      `<td class="mono dim">${esc(c.parent || "—")}</td><td style="white-space:normal;max-width:70ch">${esc(c.summary)}</td>`;
    tb.append(tr);
  }
  t.append(tb); reg.append(el("div", "scroll", "").appendChild(t).parentNode);
  v.append(reg);
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
      `<td class="n ${g.geomean_compiled === bestGeo && bestGeo > 0 ? "best" : ""}" style="color:var(--teal)">${sx(g.geomean_compiled)}</td>` +
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
    "Colour is on a LOG scale (1× → 34×); linear would flatten the whole grid into one shade. Config 14 is excluded: it is a feasibility probe, not a timed config. Each cell carries BOTH numbers — the large one is the selected baseline, the small one the other. Unmeasured cells are outlined and empty; a failed cell is hatched."));

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
  scale.append(el("span", "lab", "1×"));
  const bar = el("div", "bar");
  for (let i = 0; i <= 24; i++) {
    const sp = Math.exp((i / 24) * Math.log(TOP));
    const b = el("i"); b.style.background = rgb(heatColor(sp)); b.style.width = "14px"; bar.append(b);
  }
  scale.append(bar);
  scale.append(el("span", "lab", "34×  (log)"));
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
      `<td class="n" style="color:var(--amber)">${e && k ? sx(e / k) : "—"}</td>`;
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
      `<td class="n" style="color:var(--teal)"><b>${sx(lane.geomean_compiled)}</b></td>` +
      `<td class="n" style="color:var(--amber)">${ratio ? nf(ratio, 2) + "× flattered" : "—"}</td>`;
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
        `<td class="n" style="color:var(--teal)">${sx(best.cell.speedup_compiled)}</td>`;
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
      `<td class="n" style="color:${frac == null ? "var(--faint)" : frac >= 1 ? "var(--brick)" : "var(--amber)"}">${frac == null ? "—" : nf(frac, 2)}</td>` +
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
    tr.innerHTML =
      `<td class="n">${c.id}</td><td class="dim tiny">${esc(c.regime.replace(/_/g, " "))}</td>` +
      `<td><span class="pill ${r.passed ? "ok" : "bad"}">${esc(r.status)}</span></td>` +
      `<td class="n">${ms(r.candidate_ms)}</td><td class="n dim">${ms(r.baseline_ms ?? SNAP.baselines.eager_ms[c.id])}</td>` +
      `<td class="n dim">${ms(r.compiled_baseline_ms)}</td>` +
      `<td class="n" style="color:var(--faint)">${sx(r.speedup)}</td>` +
      `<td class="n" style="color:var(--teal)">${sx(r.speedup_compiled)}</td>` +
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
  const dl = el("dl", "kv");
  const spec = SNAP.candidates.find((c) => c.name === g.candidate);
  const add = (k, v) => { dl.append(el("dt", null, k)); dl.append(el("dd", null, v)); };
  add("commit", g.commit_sha);
  add("branch", g.branch || "—");
  add("generation", spec ? String(spec.generation) : "—");
  add("parent", spec?.parent || "—");
  add("measured / passed", `${g.configs_measured} / ${g.configs_passed}`);
  add("geomean vs eager", sx(g.geomean));
  add("geomean vs compiled", sx(g.geomean_compiled));
  add("weighted score", `${nf(g.weighted_score, 3)} eager · ${nf(g.weighted_score_compiled, 3)} compiled`);
  add("last measured", `${g.latest_ts} (${ago(g.latest_ts)} ago)`);
  b.append(dl);
  if (spec) { const p = el("p", "sub"); p.style.maxWidth = "none"; p.textContent = spec.summary; b.append(p); }
  b.append(el("h2", null, "Per config"));
  b.append(el("div", "scroll", "").appendChild(perConfigTable(g)).parentNode);
}

function openCommit(sha) {
  const gs = groupsAt(sha);
  const node = SNAP.tree.nodes.find((n) => n.sha === sha);
  $("#drawer").hidden = false;
  $("#drawer-title").textContent = `${node?.short || sha.slice(0, 8)} — ${node?.subject || ""}`.slice(0, 90);
  const b = $("#drawer-body");
  b.innerHTML = "";
  const dl = el("dl", "kv");
  const add = (k, v) => { dl.append(el("dt", null, k)); dl.append(el("dd", null, v)); };
  add("commit", sha);
  add("subject", node?.subject || "—");
  add("authored", node?.when || "—");
  add("kind", node?.isMerge ? "merge (two parents)" : "commit");
  add("branch tips", (node?.tips || []).join(", ") || "—");
  add("ledger rows", String(node?.rows ?? 0));
  b.append(dl);
  if (!gs.length) { b.append(el("p", "empty-note", "no measurements recorded against this commit")); return; }
  for (const g of gs) {
    b.append(el("h2", null, `${g.candidate} @ pad ${g.padding_ratio.toFixed(1)} — geomean ${sx(g.geomean)} eager / ${sx(g.geomean_compiled)} compiled`));
    b.append(el("div", "scroll", "").appendChild(perConfigTable(g)).parentNode);
  }
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

// Live derivations over the ratchet ledger. Everything here is READ-ONLY:
// the repo is never written to, never cached to disk, and every poll re-reads source.
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import path from "node:path";

const pexec = promisify(execFile);

export const REPO = path.resolve(process.env.RATCHET_REPO || new URL("../..", import.meta.url).pathname);
// RATCHET_RESULTS exists so the parser can be exercised against a scratch copy
// without ever touching the real append-only ledger.
export const RESULTS = process.env.RATCHET_RESULTS || path.join(REPO, "bench", "results.jsonl");
export const FINDINGS = path.join(REPO, "docs", "findings");

// The correctness budget the project locks (CLAUDE.md rule 1). Not configurable here.
export const MAX_ABS_BUDGET = 2e-3;
export const SCORE_CAP = 3.0;
const RUNNERS = "run_matrix|loop|probe_config14";
// Must look like an actual interpreter invocation, not merely a shell line that mentions
// the file (an agent typing `./scripts/run-loop.sh run_matrix.py` is not a running loop).
const RUNNER_SCRIPT_RE = new RegExp(String.raw`(?:^|/)python[\d.]*\s+(?:-\S+\s+)*(?:\S*/)?(${RUNNERS})\.py(?:\s|$)`);
const RUNNER_MODULE_RE = new RegExp(String.raw`(?:^|/)python[\d.]*\s+(?:-\S+\s+)*-m\s+(?:[\w.]+\.)?(${RUNNERS})(?:\s|$)`);
const SHELL_RE = /^(?:\S*\/)?(?:ba|z|k|da)?sh\b/;

async function git(args) {
  const { stdout } = await pexec("git", args, { cwd: REPO, maxBuffer: 16 * 1024 * 1024 });
  return stdout;
}

// ---------------------------------------------------------------------------
// results.jsonl -- append-only, fsync'd per row, so the LAST line may be torn
// mid-write. A torn tail is normal operation, not an error: skip it and move on.
// ---------------------------------------------------------------------------
export function readResults() {
  let raw;
  try {
    raw = fs.readFileSync(RESULTS, "utf8");
  } catch (err) {
    return { rows: [], lines: 0, tornTail: false, malformed: 0, error: String(err.message || err) };
  }
  const lines = raw.split("\n");
  const rows = [];
  let malformed = 0;
  let tornTail = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch {
      // Only the final populated line can legitimately be a partial write.
      const rest = lines.slice(i + 1).join("").trim();
      if (rest === "") tornTail = true;
      else malformed++;
    }
  }
  return { rows, lines: rows.length, tornTail, malformed, error: null };
}

// Mirrors bench/ledger.py::_padding_of exactly. Rows predating the field were all
// measured unpadded, so 0.0 is history, not a guess.
export function paddingOf(row) {
  if (row.padding_ratio !== undefined && row.padding_ratio !== null) return Number(row.padding_ratio);
  const note = row.notes || "";
  const at = note.indexOf("padding_ratio=");
  if (at >= 0) {
    const v = parseFloat(note.slice(at + "padding_ratio=".length).split(/\s/)[0]);
    if (Number.isFinite(v)) return v;
  }
  return 0.0;
}

const isClean = (r) => r.dirty !== true;
const isOk = (r) => r.status === "ok" && !!(r.correctness && r.correctness.passed);

function geomean(values) {
  const v = values.filter((x) => Number.isFinite(x) && x > 0);
  if (!v.length) return null;
  return Math.exp(v.reduce((a, b) => a + Math.log(b), 0) / v.length);
}

// bench/matrix.py::weighted_score -- clipped, and a config with no entry scores 1.0
// rather than being skipped, so not measuring is never rewarded.
function weightedScore(speedupsByConfig, matrixIds) {
  const keys = Object.keys(speedupsByConfig);
  if (!keys.length) return 0.0;
  let total = 0;
  for (const id of matrixIds) total += Math.min(speedupsByConfig[id] ?? 1.0, SCORE_CAP);
  return total / matrixIds.length;
}

// ---------------------------------------------------------------------------
// Static, cached once at server start: the matrix and the candidate registry.
// ---------------------------------------------------------------------------
const PY_MATRIX = `import sys;sys.path.insert(0,${JSON.stringify(REPO)});import json;from bench.matrix import MATRIX,regime_of;print(json.dumps([{**c.to_dict(),'regime':regime_of(c.id)} for c in MATRIX]))`;
// The DECLARED lineage -- the registry's parent graph, which is the real evolutionary
// tree (finding 28). Its derivation needs more than one line (recombination edges,
// known-unsafe and topology-violation sets), so it lives in a file rather than -c.
const PY_LINEAGE = path.join(REPO, "dashboard", "server", "lineage.py");

async function pyJson(code) {
  const { stdout } = await pexec("python3", ["-c", code], { cwd: REPO, maxBuffer: 8 * 1024 * 1024 });
  return JSON.parse(stdout);
}

async function pyScript(file) {
  const { stdout } = await pexec("python3", [file], { cwd: REPO, maxBuffer: 8 * 1024 * 1024 });
  return JSON.parse(stdout);
}

let STATIC = null;
let CAND_AT = 0;
const CAND_TTL_MS = 30000;
const EMPTY_LINEAGE = { nodes: [], recombination_edges: [], known_unsafe: [], topology_violations: [] };

export async function loadStatic() {
  if (!STATIC) {
    STATIC = { matrix: [], candidates: [], lineage: EMPTY_LINEAGE, errors: [] };
    // The matrix is the 14 announced competition configs: genuinely static, read once.
    try { STATIC.matrix = await pyJson(PY_MATRIX); }
    catch (e) { STATIC.errors.push("matrix: " + (e.stderr || e.message)); }
  }
  // The candidate registry GROWS while the loop runs, so it is refreshed on a slow
  // timer rather than pinned at server start.
  if (Date.now() - CAND_AT > CAND_TTL_MS) {
    CAND_AT = Date.now();
    try {
      const lineage = await pyScript(PY_LINEAGE);
      STATIC.lineage = lineage;
      // `candidates` is the flat registry view the scoreboard/heatmap/drawer already
      // read; it is now a projection of the lineage rather than a second query.
      STATIC.candidates = lineage.nodes.map(
        ({ name, generation, parent, summary }) => ({ name, generation, parent, summary }));
      STATIC.errors = STATIC.errors.filter((e) => !e.startsWith("candidates:"));
    } catch (e) {
      if (!STATIC.candidates.length) STATIC.errors.push("candidates: " + (e.stderr || e.message));
    }
  }
  return STATIC;
}

// ---------------------------------------------------------------------------
// git: the commit DAG. NOT the evolutionary tree -- finding 28 measured that every
// candidate has exactly `generation - 1` git ancestors, because every candidate branch
// was cut from `ben`'s tip and every candidate is merged back into `ben`. The spurs in
// `git log --graph` are decorative. This is kept because it is the real record of what
// was committed when; the lineage the dashboard DRAWS comes from the registry
// (see lineage.py and `snapshot().lineage`).
// ---------------------------------------------------------------------------
async function gitState() {
  const state = { commits: [], branches: [], head: null, dirty: null, dirtyFiles: [], error: null };
  try {
    const log = await git(["log", "--format=%H|%h|%s|%P|%an|%aI", "--all"]);
    for (const line of log.split("\n")) {
      if (!line.trim()) continue;
      const [sha, short, subject, parents, author, when] = line.split("|");
      state.commits.push({
        sha, short, subject, author, when,
        parents: (parents || "").trim() ? parents.trim().split(/\s+/) : [],
      });
    }
    const br = await git(["branch", "-a"]);
    for (const line of br.split("\n")) {
      if (!line.trim()) continue;
      const current = line.startsWith("*");
      const name = line.replace(/^[*+]?\s*/, "").trim();
      if (name.includes("->")) continue; // symbolic ref like origin/HEAD
      state.branches.push({ name, current, remote: name.startsWith("remotes/") });
      if (current) state.head = { branch: name };
    }
    const porcelain = await git(["status", "--porcelain"]);
    state.dirtyFiles = porcelain.split("\n").filter((l) => l.trim()).map((l) => l.trim());
    state.dirty = state.dirtyFiles.length > 0;
    const headSha = (await git(["rev-parse", "HEAD"])).trim();
    state.head = { branch: state.head?.branch ?? "(detached)", sha: headSha, short: headSha.slice(0, 8) };
  } catch (e) {
    state.error = String(e.stderr || e.message || e);
  }
  return state;
}

// Reduce the full 60+ commit history to the nodes that carry meaning: commits with
// measurements, merges, merge parents, and branch tips. Edges are nearest-interesting
// ancestry, so the diamond around the merge survives the reduction. This is git
// TOPOLOGY -- documentation of the commit record, not the lineage mechanism.
function buildTree(gitS, rows) {
  const byShaCommit = new Map(gitS.commits.map((c) => [c.sha, c]));
  const measured = new Map(); // sha -> {candidates:Set, rows:n}
  for (const r of rows) {
    if (!r.commit_sha) continue;
    if (!measured.has(r.commit_sha)) measured.set(r.commit_sha, { candidates: new Set(), rows: 0 });
    const m = measured.get(r.commit_sha);
    m.rows++;
    if (r.candidate) m.candidates.add(r.candidate);
  }

  const interesting = new Set();
  for (const sha of measured.keys()) if (byShaCommit.has(sha)) interesting.add(sha);
  for (const c of gitS.commits) {
    if (c.parents.length > 1) { interesting.add(c.sha); for (const p of c.parents) interesting.add(p); }
  }
  for (const sha of gitS.branchTips || []) interesting.add(sha);

  // nearest interesting ancestors, walking through uninteresting commits
  const nearest = (sha, seen = new Set()) => {
    const out = new Set();
    const stack = [...(byShaCommit.get(sha)?.parents || [])];
    while (stack.length) {
      const p = stack.pop();
      if (seen.has(p)) continue;
      seen.add(p);
      if (interesting.has(p)) out.add(p);
      else stack.push(...(byShaCommit.get(p)?.parents || []));
    }
    return out;
  };

  const nodes = [];
  const edges = [];
  for (const sha of interesting) {
    const c = byShaCommit.get(sha);
    if (!c) continue;
    const m = measured.get(sha);
    nodes.push({
      sha,
      short: c.short,
      subject: c.subject,
      when: c.when,
      isMerge: c.parents.length > 1,
      rows: m ? m.rows : 0,
      candidates: m ? [...m.candidates] : [],
      tips: (gitS.tipsBySha && gitS.tipsBySha[sha]) || [],
    });
    for (const p of nearest(sha)) edges.push({ from: p, to: sha });
  }

  // Transitive reduction: drop an edge a->b when b is also reachable from a
  // through another kept edge. Keeps the diamond, kills the shortcut clutter.
  const adj = new Map(nodes.map((n) => [n.sha, []]));
  for (const e of edges) adj.get(e.from)?.push(e.to);
  const kept = edges.filter((e) => {
    const others = (adj.get(e.from) || []).filter((t) => t !== e.to);
    for (const o of others) {
      const seen = new Set(); const st = [o];
      while (st.length) { const x = st.pop(); if (seen.has(x)) continue; seen.add(x); st.push(...(adj.get(x) || [])); }
      if (seen.has(e.to)) return false;
    }
    return true;
  });

  // longest-path depth for layering
  const depth = new Map();
  const parentsOf = new Map(nodes.map((n) => [n.sha, []]));
  for (const e of kept) parentsOf.get(e.to)?.push(e.from);
  const resolve = (sha, guard = new Set()) => {
    if (depth.has(sha)) return depth.get(sha);
    if (guard.has(sha)) return 0;
    guard.add(sha);
    const ps = parentsOf.get(sha) || [];
    const d = ps.length ? Math.max(...ps.map((p) => resolve(p, guard))) + 1 : 0;
    depth.set(sha, d);
    return d;
  };
  for (const n of nodes) n.depth = resolve(n.sha);
  nodes.sort((a, b) => a.depth - b.depth || (a.when < b.when ? -1 : 1));
  return { nodes, edges: kept };
}

// ---------------------------------------------------------------------------
// Is anything running right now?
// ---------------------------------------------------------------------------
async function runningProcesses() {
  try {
    const { stdout } = await pexec("ps", ["-eo", "pid=,etimes=,args="], { maxBuffer: 8 * 1024 * 1024 });
    const hits = [];
    for (const line of stdout.split("\n")) {
      const m = line.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/);
      if (!m) continue;
      const [, pid, etimes, args] = m;
      if (Number(pid) === process.pid) continue;
      if (SHELL_RE.test(args.trim())) continue;          // shell wrappers, not the runner
      const hit = RUNNER_SCRIPT_RE.exec(args) || RUNNER_MODULE_RE.exec(args);
      if (!hit) continue;
      hits.push({ pid: Number(pid), elapsed_s: Number(etimes), runner: hit[1] + ".py", cmd: args.slice(0, 220) });
    }
    return hits;
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// docs/findings
// ---------------------------------------------------------------------------
function readFindings() {
  const out = { learnings: null, docs: [], error: null };
  try {
    const names = fs.readdirSync(FINDINGS).filter((n) => /^\d\d-.*\.md$/.test(n)).sort();
    for (const n of names) {
      const p = path.join(FINDINGS, n);
      const text = fs.readFileSync(p, "utf8");
      const stat = fs.statSync(p);
      const title = (text.split("\n").find((l) => l.startsWith("# ")) || n).replace(/^#\s*/, "");
      const doc = { file: n, title, mtime: stat.mtimeMs, bytes: stat.size, text };
      if (n.startsWith("00-")) out.learnings = doc;
      out.docs.push({ file: n, title, mtime: stat.mtimeMs, bytes: stat.size });
    }
    if (out.learnings) out.learningsText = out.learnings.text;
  } catch (e) {
    out.error = String(e.message || e);
  }
  return out;
}

export function readFindingDoc(file) {
  if (!/^\d\d-[A-Za-z0-9._-]+\.md$/.test(file)) return null;
  const p = path.join(FINDINGS, file);
  if (!p.startsWith(FINDINGS)) return null;
  try { return fs.readFileSync(p, "utf8"); } catch { return null; }
}

// ---------------------------------------------------------------------------
// The snapshot the UI renders.
// ---------------------------------------------------------------------------
export async function snapshot() {
  const stat = await loadStatic();
  const matrix = stat.matrix;
  const matrixIds = matrix.map((c) => c.id);
  const runnable = matrix.filter((c) => c.id !== 14).map((c) => c.id); // 14 is a feasibility probe

  const parsed = readResults();
  const rows = parsed.rows;
  const gitS = await gitState();

  // branch tips, for tree annotation
  gitS.tipsBySha = {};
  try {
    const showRef = await git(["show-ref", "--heads"]);
    for (const line of showRef.split("\n")) {
      if (!line.trim()) continue;
      const [sha, ref] = line.trim().split(/\s+/);
      const name = ref.replace("refs/heads/", "");
      (gitS.tipsBySha[sha] ||= []).push(name);
    }
  } catch { /* no refs -> no annotation */ }
  gitS.branchTips = Object.keys(gitS.tipsBySha);

  // ---- baselines -------------------------------------------------------
  // Two of them. Ledger speedups are computed against EAGER; the honest number
  // is against torch.compile(max-autotune). Both are surfaced, never just one.
  const pickLatest = (pred) => {
    const m = new Map();
    for (const r of rows) {
      if (!pred(r)) continue;
      const prev = m.get(r.config_id);
      if (!prev || String(r.ts) > String(prev.ts)) m.set(r.config_id, r);
    }
    return m;
  };
  const eagerRows = pickLatest((r) => r.candidate === "baseline" && r.timing);
  const compiledRows = pickLatest((r) => r.candidate === "baseline_compiled" && r.timing);
  const eagerMs = {}, compiledMs = {};
  for (const [cid, r] of eagerRows) eagerMs[cid] = r.timing.candidate_ms;
  for (const [cid, r] of compiledRows) compiledMs[cid] = r.timing.candidate_ms;

  const vsCompiled = (r) => {
    const base = compiledMs[r.config_id];
    const ms = r.timing?.candidate_ms;
    if (!Number.isFinite(base) || !Number.isFinite(ms) || ms <= 0) return null;
    return base / ms;
  };

  // ---- scoreboard: keyed (commit, candidate, padding_ratio). NEVER pooled. --
  const groups = new Map();
  for (const r of rows) {
    if (!isClean(r)) continue;
    const pad = paddingOf(r);
    const key = `${r.commit_sha}|${r.candidate || ""}|${pad}`;
    if (!groups.has(key)) {
      groups.set(key, {
        key, commit_sha: r.commit_sha, short_sha: (r.commit_sha || "").slice(0, 8),
        candidate: r.candidate || "", padding_ratio: pad, branch: r.branch,
        rows: 0, seen: new Set(), seen_passed: new Set(),
        speedups: {}, speedups_compiled: {}, per_config: {}, failures: [],
        latest_ts: r.ts,
      });
    }
    const g = groups.get(key);
    // Rows, not configs: a parameter sweep writes many rows for the same config at one
    // commit. The matrix has 14 configs, so "configs measured" must be DISTINCT ids,
    // last-write-wins per config, with the raw row count kept alongside it.
    g.rows++;
    g.seen.add(r.config_id);
    if (String(r.ts) > String(g.latest_ts)) g.latest_ts = r.ts;
    const ok = isOk(r);
    const spc = vsCompiled(r);
    g.per_config[r.config_id] = {
      config_id: r.config_id, status: r.status, passed: ok,
      speedup: r.timing?.speedup ?? null, speedup_compiled: spc,
      candidate_ms: r.timing?.candidate_ms ?? null,
      baseline_ms: r.timing?.baseline_ms ?? null,
      compiled_baseline_ms: compiledMs[r.config_id] ?? null,
      method: r.timing?.method ?? null, samples: r.timing?.samples ?? null,
      interleaved: r.timing?.interleaved ?? null, arms_isolated: r.timing?.arms_isolated ?? null,
      max_abs: r.correctness?.max_abs ?? null, max_rel: r.correctness?.max_rel ?? null,
      failed_elements: r.correctness?.failed_elements ?? null,
      peak_MB: r.memory?.peak_MB ?? null, ts: r.ts,
    };
    if (ok) {
      g.seen_passed.add(r.config_id);
      if (Number.isFinite(r.timing?.speedup)) g.speedups[r.config_id] = r.timing.speedup;
      if (Number.isFinite(spc)) g.speedups_compiled[r.config_id] = spc;
    } else {
      g.seen_passed.delete(r.config_id);
      g.failures.push({ config_id: r.config_id, status: r.status });
    }
  }
  const scoreboard = [...groups.values()].map(({ seen, seen_passed, ...g }) => ({
    ...g,
    configs_measured: seen.size,
    configs_passed: seen_passed.size,
    sweep: g.rows > seen.size,
    // last-write-wins per config, so the failure list is derived from the final state
    failures: Object.values(g.per_config).filter((c) => !c.passed)
      .map((c) => ({ config_id: c.config_id, status: c.status })),
    geomean: geomean(Object.values(g.speedups)),
    geomean_compiled: geomean(Object.values(g.speedups_compiled)),
    weighted_score: weightedScore(g.speedups, matrixIds),
    weighted_score_compiled: weightedScore(g.speedups_compiled, matrixIds),
  })).sort((a, b) => b.weighted_score - a.weighted_score);

  // ---- heatmap: one lane per (candidate, padding), newest commit wins -------
  // One lane per (candidate, padding). Where a candidate has several groups at that
  // padding -- a full matrix sweep plus a one-config re-probe, say -- the lane shows the
  // MOST COMPLETE run, then the most recent, so a single follow-up row cannot blank out
  // an otherwise fully measured candidate.
  const laneBest = new Map();
  for (const g of scoreboard) {
    if (g.candidate === "baseline") continue;
    const lane = `${g.candidate}|${g.padding_ratio}`;
    const prev = laneBest.get(lane);
    const better = !prev
      || g.configs_measured > prev.configs_measured
      || (g.configs_measured === prev.configs_measured && String(g.latest_ts) > String(prev.latest_ts));
    if (better) laneBest.set(lane, g);
  }
  const regByName = new Map(stat.candidates.map((c) => [c.name, c]));
  const heatmap = {
    configs: runnable,
    lanes: [...laneBest.values()]
      .map((g) => ({
        candidate: g.candidate, padding_ratio: g.padding_ratio,
        commit_sha: g.commit_sha, short_sha: g.short_sha,
        generation: regByName.get(g.candidate)?.generation ?? null,
        geomean: g.geomean, geomean_compiled: g.geomean_compiled,
        cells: Object.fromEntries(runnable.map((cid) => [cid, g.per_config[cid] ?? null])),
      }))
      .sort((a, b) => (a.generation ?? 99) - (b.generation ?? 99)
        || a.candidate.localeCompare(b.candidate)
        || a.padding_ratio - b.padding_ratio),
  };

  // ---- failures: results, not errors to hide -------------------------------
  const failures = rows
    .filter((r) => r.status !== "ok" || (r.correctness && r.correctness.passed === false))
    .map((r) => ({
      ts: r.ts, candidate: r.candidate, config_id: r.config_id, branch: r.branch,
      commit_short: (r.commit_sha || "").slice(0, 8), dirty: !!r.dirty,
      padding_ratio: paddingOf(r), status: r.status,
      max_abs: r.correctness?.max_abs ?? null, max_rel: r.correctness?.max_rel ?? null,
      failed_elements: r.correctness?.failed_elements ?? null,
      budget: MAX_ABS_BUDGET,
      notes: (r.notes || "").slice(0, 400),
    }))
    .sort((a, b) => (a.ts < b.ts ? 1 : -1));

  // Newest by parsed instant, not string order: rows may carry different UTC offsets.
  // Some seed rows are stamped in the future; that is reported, not silently clamped.
  const nowMs = Date.now();
  let newest = null, newestMs = -Infinity;
  let newestPast = null, newestPastMs = -Infinity, futureRows = 0;
  for (const r of rows) {
    const t = Date.parse(r.ts);
    if (!Number.isFinite(t)) continue;
    if (t > newestMs) { newestMs = t; newest = r.ts; }
    if (t > nowMs + 2000) { futureRows++; continue; }
    if (t > newestPastMs) { newestPastMs = t; newestPast = r.ts; }
  }
  const running = await runningProcesses();
  const findings = readFindings();

  let fileStat = null;
  try { const s = fs.statSync(RESULTS); fileStat = { bytes: s.size, mtime: s.mtimeMs }; } catch {}

  return {
    generated_at: new Date().toISOString(),
    repo: REPO,
    results_path: RESULTS,
    parse: { rows: parsed.lines, torn_tail: parsed.tornTail, malformed: parsed.malformed, error: parsed.error },
    file: fileStat,
    newest_row_ts: newest,
    newest_past_row_ts: newestPast,
    future_rows: futureRows,
    running,
    git: {
      head: gitS.head, dirty: gitS.dirty, dirty_files: gitS.dirtyFiles.slice(0, 40),
      branches: gitS.branches, error: gitS.error,
    },
    matrix, matrix_static_errors: stat.errors,
    candidates: stat.candidates,
    // THE tree: the declared-parent graph from the registry (finding 28).
    lineage: stat.lineage,
    // Git topology, retained as documentation. It is a chain, not the lineage.
    tree: buildTree(gitS, rows),
    baselines: {
      eager_ms: eagerMs, compiled_ms: compiledMs,
      eager_rows: eagerRows.size, compiled_rows: compiledRows.size,
      compiled_speedup_vs_eager: Object.fromEntries(
        Object.keys(compiledMs).filter((c) => eagerMs[c]).map((c) => [c, eagerMs[c] / compiledMs[c]])),
    },
    scoreboard, heatmap, failures,
    findings: { docs: findings.docs, learnings: findings.learnings?.text ?? null, error: findings.error },
    counts: {
      rows: rows.length, clean: rows.filter(isClean).length,
      dirty: rows.filter((r) => !isClean(r)).length,
      ok: rows.filter((r) => r.status === "ok").length,
      candidates: new Set(rows.map((r) => r.candidate)).size,
      commits: new Set(rows.map((r) => r.commit_sha)).size,
    },
  };
}

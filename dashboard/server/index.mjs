// Live dashboard server for the ratchet autoresearch loop.
// Zero dependencies. Binds loopback only. Read-only against the repo.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { snapshot, loadStatic, readFindingDoc, REPO, RESULTS } from "./data.mjs";

const HOST = "127.0.0.1";                       // loopback only, deliberately
const BASE_PORT = Number(process.env.PORT || 5177);
const POLL_MS = Number(process.env.POLL_MS || 2000);
const PUBLIC = path.resolve(new URL("../public", import.meta.url).pathname);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json; charset=utf-8",
};

// --- one shared poll loop; N browser tabs must not mean N times the work ----
const clients = new Set();
let latest = null;
let latestJson = "";
let lastCore = "";
let seq = 0;
let polling = false;

async function poll() {
  if (polling) return;
  polling = true;
  try {
    const snap = await snapshot();
    // The wall-clock stamp changes every poll; hashing it would make every poll
    // look like a change and turn "push on change" into a 2 s broadcast.
    const stamp = snap.generated_at;
    snap.generated_at = "";
    const core = JSON.stringify(snap);
    snap.generated_at = stamp;
    if (core !== lastCore) {
      lastCore = core;
      latest = snap;
      latestJson = JSON.stringify(snap);
      seq++;
      const json = latestJson;
      const frame = `event: snapshot\ndata: ${json}\nid: ${seq}\n\n`;
      for (const res of clients) { try { res.write(frame); } catch { clients.delete(res); } }
    } else if (latest) {
      latest.generated_at = stamp;   // /api/snapshot still reports real freshness
    }
  } catch (err) {
    const frame = `event: error\ndata: ${JSON.stringify({ error: String(err.message || err) })}\n\n`;
    for (const res of clients) { try { res.write(frame); } catch { clients.delete(res); } }
    process.stderr.write(`[poll] ${err.stack || err}\n`);
  } finally {
    polling = false;
  }
}

function serveStatic(req, res, urlPath) {
  const rel = urlPath === "/" ? "/index.html" : urlPath;
  const file = path.join(PUBLIC, path.normalize(rel).replace(/^(\.\.[/\\])+/, ""));
  if (!file.startsWith(PUBLIC)) { res.writeHead(403).end("forbidden"); return; }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404, { "content-type": "text/plain" }).end("not found"); return; }
    res.writeHead(200, {
      "content-type": MIME[path.extname(file)] || "application/octet-stream",
      "cache-control": "no-store",
    }).end(buf);
  });
}

const json = (res, code, obj) =>
  res.writeHead(code, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" })
     .end(JSON.stringify(obj));

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}`);
  const p = url.pathname;

  if (p === "/api/snapshot") {
    try {
      if (!latest) await poll();
      return json(res, 200, latest ?? { error: "no snapshot yet" });
    } catch (e) { return json(res, 500, { error: String(e.message || e) }); }
  }

  if (p === "/api/health") {
    return json(res, 200, {
      ok: true, repo: REPO, results: RESULTS, poll_ms: POLL_MS,
      clients: clients.size, seq, rows: latest?.counts?.rows ?? null,
    });
  }

  if (p === "/api/finding") {
    const text = readFindingDoc(url.searchParams.get("file") || "");
    if (text == null) return json(res, 404, { error: "unknown finding" });
    return json(res, 200, { file: url.searchParams.get("file"), text });
  }

  if (p === "/api/stream") {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    res.write("retry: 3000\n\n");
    clients.add(res);
    if (!latest) await poll();
    if (latestJson) res.write(`event: snapshot\ndata: ${latestJson}\nid: ${seq}\n\n`);
    const beat = setInterval(() => { try { res.write(": ping\n\n"); } catch {} }, 15000);
    req.on("close", () => { clearInterval(beat); clients.delete(res); });
    return;
  }

  if (p.startsWith("/api/")) return json(res, 404, { error: "unknown endpoint" });
  return serveStatic(req, res, p);
});

function listen(port, attemptsLeft = 12) {
  server.once("error", (err) => {
    if (err.code === "EADDRINUSE" && attemptsLeft > 0) {
      process.stderr.write(`port ${port} busy, trying ${port + 1}\n`);
      listen(port + 1, attemptsLeft - 1);
    } else { throw err; }
  });
  server.listen(port, HOST, () => {
    process.stdout.write(
      `\nratchet dashboard  ->  http://${HOST}:${port}\n` +
      `  repo    ${REPO}\n  ledger  ${RESULTS}\n  poll    every ${POLL_MS} ms (SSE push on change)\n\n`);
  });
}

await loadStatic();
await poll();
setInterval(poll, POLL_MS);
listen(BASE_PORT);

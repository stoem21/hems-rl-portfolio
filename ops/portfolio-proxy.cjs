'use strict';

// Dependency-free API gateway. Static frontends are deployed separately to Pages.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');

function loadConfig(filename) {
  const config = JSON.parse(fs.readFileSync(filename, 'utf8'));
  if (config.schemaVersion !== 1) throw new Error('Unsupported configuration version');
  for (const key of ['proxyPort', 'clinicPort', 'hemsPort', 'maxHemsJobs', 'maxBodyBytes', 'upstreamTimeoutMs']) {
    if (!Number.isInteger(config[key]) || config[key] < 1) throw new Error(`Invalid ${key}`);
  }
  for (const key of ['proxyPort', 'clinicPort', 'hemsPort']) {
    if (config[key] > 65535) throw new Error(`Invalid ${key}`);
  }
  if (new Set([config.proxyPort, config.clinicPort, config.hemsPort]).size !== 3) {
    throw new Error('Proxy and backend ports must be different');
  }
  for (const key of ['clinicDemoDir', 'hemsProjectDir']) {
    if (typeof config[key] !== 'string' || !path.isAbsolute(config[key])) throw new Error(`Invalid ${key}`);
  }
  if (config.hemsDashboardDataFile !== undefined &&
      (typeof config.hemsDashboardDataFile !== 'string' || !path.isAbsolute(config.hemsDashboardDataFile))) {
    throw new Error('Invalid hemsDashboardDataFile');
  }
  if (!Array.isArray(config.allowedOrigins) || config.allowedOrigins.length === 0) throw new Error('Set allowedOrigins');
  for (const origin of config.allowedOrigins) {
    const parsed = new URL(origin);
    if (!['http:', 'https:'].includes(parsed.protocol) || parsed.origin !== origin) {
      throw new Error('Origins must contain only scheme, hostname and optional port; no path or trailing slash');
    }
  }
  if (typeof config.publicUrl !== 'string') throw new Error('Invalid publicUrl');
  if (config.publicUrl) {
    const parsed = new URL(config.publicUrl);
    if (parsed.protocol !== 'https:' || parsed.origin !== config.publicUrl) throw new Error('publicUrl must be an HTTPS origin');
  }
  return config;
}

// Only a trusted, host-configured evaluation export is read. No request selects files.
const DASHBOARD_KEYS = ["ALGO_MEANS", "SAVING_PCT", "HH_DATA", "CLUSTER_DATA", "ROBUSTNESS", "NO_ERR_MEANS", "STRATEGY_COLORS", "STRATEGY_LABELS", "ALL_ALGOS", "ERR_ALGOS", "SIM_ALGOS", "ERROR_TYPES", "ERROR_PCTS", "DATE_DATASET_START", "DATE_TRAIN_END", "DATE_VAL_END", "DATE_TEST_START", "SEEDS", "HOUSEHOLDS", "CLUSTERS", "TRAIN_HOUSEHOLDS", "TEST_HOUSEHOLDS", "ALL_HOUSEHOLDS", "HH_META", "SEED_HH_DATA", "ROBUSTNESS_BY_HH"];
function createDashboardLoader(filename) {
  let cached;
  return async () => {
    if (!filename) throw new Error('Dashboard export is not configured');
    const bytes = await fs.promises.readFile(filename);
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    if (cached?.source.sha256 === sha256) return cached;
    const text = bytes.toString('utf8');
    if (/FICTIONAL|fictional sample data/.test(text)) throw new Error('Sample data is not an evaluation export');
    const stat = await fs.promises.stat(filename);
    // JS is an operator-owned generated module, never user-supplied executable input.
    const exported = filename.endsWith('.json') ? JSON.parse(text) :
      await import(`${pathToFileURL(filename).href}?sha256=${sha256}`);
    const data = {};
    for (const key of DASHBOARD_KEYS) {
      if (exported[key] === undefined || exported[key] === null) throw new Error(`Missing dashboard field: ${key}`);
      data[key] = exported[key];
    }
    if (!Array.isArray(data.ALL_ALGOS) || !data.ALL_ALGOS.length ||
        !data.ALL_ALGOS.every(key => Number.isFinite(data.ALGO_MEANS[key]) && Number.isFinite(data.SAVING_PCT[key]))) {
      throw new Error('Invalid evaluation metrics');
    }
    cached = { schemaVersion: 1, data, source: { kind: 'saved-evaluation',
      updatedAt: stat.mtime.toISOString(), sha256 } };
    return cached;
  };
}

const HOP_HEADERS = new Set(['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade']);

function cleanHeaders(headers) {
  const result = { ...headers };
  const connectionTokens = String(headers.connection || '').split(',').map(value => value.trim().toLowerCase());
  for (const name of Object.keys(result)) {
    if (HOP_HEADERS.has(name) || connectionTokens.includes(name) || name.startsWith('access-control-')) delete result[name];
  }
  return result;
}

function resolveRoute(rawUrl, config) {
  const index = rawUrl.indexOf('?');
  const pathname = index === -1 ? rawUrl : rawUrl.slice(0, index);
  if (pathname === '/hems/dashboard-data' || pathname === '/hems/dashboard-data/') return { dashboard: true };
  if (pathname === '/clinic/api' || pathname.startsWith('/clinic/api/')) {
    return { prefix: '/clinic', port: config.clinicPort, path: rawUrl.slice(7), heavy: false };
  }
  const hemsPaths = new Set(['/health', '/models', '/household-range', '/simulate/validate',
    '/simulate', '/simulate/compare', '/ask', '/explain']);
  if (pathname.startsWith('/hems/')) {
    const upstreamPath = pathname.slice(5);
    if (hemsPaths.has(upstreamPath.replace(/\/$/, ''))) {
      return { prefix: '/hems', port: config.hemsPort, path: rawUrl.slice(5),
        heavy: ['/simulate', '/simulate/compare', '/ask', '/explain'].includes(upstreamPath.replace(/\/$/, '')) };
    }
  }
  return null;
}

function createPortfolioServer(config) {
  const loadDashboard = createDashboardLoader(config.hemsDashboardDataFile);
  let activeHemsJobs = 0;
  const connections = new Set();
  const outgoing = new Set();
  const allowedHeaders = ['content-type', 'authorization', 'ngrok-skip-browser-warning'];
  const allowedMethods = ['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'];

  const server = http.createServer((req, res) => {
    const origin = req.headers.origin;
    const allowedOrigin = origin && config.allowedOrigins.includes(origin);
    const cors = allowedOrigin ? {
      'access-control-allow-origin': origin,
      'access-control-allow-credentials': 'true',
      'access-control-expose-headers': 'X-Explain-Source, X-Ask-Source, Retry-After',
    } : {};
    function send(status, data, extra = {}) {
      if (res.destroyed || res.headersSent) return;
      res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store',
        vary: 'Origin', ...cors, ...extra });
      res.end(JSON.stringify(data));
    }
    if (origin && !allowedOrigin) return send(403, { error: 'Origin is not allowed. Add the exact website origin to portfolio-ngrok.json.' });
    if (!allowedMethods.includes(req.method)) return send(405, { error: 'Method not supported' }, { allow: allowedMethods.join(', ') });
    if (req.url === '/_proxy/health' && ['GET', 'HEAD'].includes(req.method)) {
      return send(200, { status: 'ok', service: 'portfolio-ngrok-proxy', schemaVersion: 1, activeHemsJobs });
    }
    const route = resolveRoute(req.url, config);
    if (!route) return send(404, { error: 'API route not found', clinic: '/clinic/api/health', hems: '/hems/health' });
    if (req.method === 'OPTIONS') {
      const method = req.headers['access-control-request-method'];
      const headers = String(req.headers['access-control-request-headers'] || '').split(',')
        .map(header => header.trim().toLowerCase()).filter(Boolean);
      if (!allowedOrigin || !allowedMethods.includes(method) || headers.some(header => !allowedHeaders.includes(header))) {
        return send(403, { error: 'CORS preflight rejected' });
      }
      res.writeHead(204, { ...cors, 'access-control-allow-methods': allowedMethods.join(', '),
        'access-control-allow-headers': allowedHeaders.join(', '), 'access-control-max-age': '600',
        vary: 'Origin, Access-Control-Request-Method, Access-Control-Request-Headers' });
      return res.end();
    }
    if (route.dashboard) {
      req.resume();
      if (!['GET', 'HEAD'].includes(req.method)) return send(405, { error: 'Use GET for dashboard data' }, { allow: 'GET, HEAD, OPTIONS' });
      loadDashboard().then(payload => {
        if (req.method === 'HEAD') {
          if (!res.destroyed) { res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store', vary: 'Origin', ...cors }); res.end(); }
        } else send(200, payload);
      }).catch(() => send(503, { error: 'Saved evaluation results are unavailable. Check hemsDashboardDataFile on the backend host.' }));
      return;
    }
    const declaredLength = Number(req.headers['content-length'] || 0);
    if (declaredLength > config.maxBodyBytes) return send(413, { error: 'Request body too large' });
    const heavy = route.heavy && req.method === 'POST';
    if (heavy && activeHemsJobs >= config.maxHemsJobs) {
      return send(429, { error: 'HEMS is busy. Please try again after the current job finishes.' }, { 'retry-after': '10' });
    }
    if (heavy) activeHemsJobs++;
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      if (heavy) activeHemsJobs--;
    };
    let size = 0;
    let rejected = false;
    const chunks = [];
    req.on('error', release);
    req.on('aborted', release);
    req.on('data', chunk => {
      if (rejected) return;
      size += chunk.length;
      if (size > config.maxBodyBytes) {
        rejected = true;
        chunks.length = 0;
        release();
        send(413, { error: 'Request body too large' }, { connection: 'close' });
      } else chunks.push(chunk);
    });
    req.on('end', () => {
      if (rejected || res.destroyed) return release();
      const headers = cleanHeaders(req.headers);
      delete headers['ngrok-skip-browser-warning'];
      delete headers['cf-connecting-ip'];
      headers.host = `127.0.0.1:${route.port}`;
      headers['x-forwarded-for'] = req.socket.remoteAddress;
      headers['x-forwarded-prefix'] = route.prefix;
      headers['content-length'] = String(size);
      const upstream = http.request({ hostname: '127.0.0.1', port: route.port,
        path: route.path, method: req.method, headers });
      outgoing.add(upstream);
      const deadline = setTimeout(() => {
        send(504, { error: 'Backend request timed out' });
        upstream.destroy(new Error('Backend timeout'));
      }, config.upstreamTimeoutMs);
      const finish = () => { clearTimeout(deadline); outgoing.delete(upstream); release(); };
      upstream.on('error', () => {
        send(502, { error: 'Backend is unavailable', service: route.prefix.slice(1) });
        if (res.headersSent && !res.writableEnded) res.destroy();
        finish();
      });
      upstream.on('response', response => {
        const responseHeaders = cleanHeaders(response.headers);
        if (responseHeaders.location) {
          try {
            const location = new URL(responseHeaders.location, `http://127.0.0.1:${route.port}${route.path}`);
            if (['127.0.0.1', 'localhost'].includes(location.hostname) && Number(location.port || 80) === route.port) {
              responseHeaders.location = route.prefix + location.pathname + location.search + location.hash;
            }
          } catch { /* Preserve an upstream Location that is not a valid URL. */ }
        }
        responseHeaders.vary = [...new Set(String(responseHeaders.vary || '').split(',').map(v => v.trim()).filter(Boolean).concat('Origin'))].join(', ');
        Object.assign(responseHeaders, cors, { 'cache-control': 'no-store' });
        response.on('end', finish);
        response.on('error', () => { res.destroy(); finish(); });
        response.on('aborted', () => { res.destroy(); finish(); });
        if (res.destroyed) return response.resume();
        res.writeHead(response.statusCode, responseHeaders);
        res.flushHeaders();
        response.pipe(res);
        res.on('close', () => {
          if (!response.complete && heavy) {
            // FastAPI threads may keep computing after a client disconnects.
            // Drain the response and keep the job slot until the upstream ends.
            response.unpipe(res);
            response.resume();
          } else if (!response.complete) upstream.destroy();
        });
      });
      res.on('close', () => { if (!heavy) upstream.destroy(); });
      upstream.end(Buffer.concat(chunks, size));
    });
  });
  server.on('connection', socket => { connections.add(socket); socket.on('close', () => connections.delete(socket)); });
  server.requestTimeout = 30000;
  server.headersTimeout = 15000;
  server.stop = () => {
    server.close();
    for (const request of outgoing) request.destroy();
    for (const socket of connections) socket.destroy();
  };
  return server;
}

if (require.main === module) {
  try {
    const config = loadConfig(process.argv[2] || path.join(__dirname, 'portfolio-ngrok.json'));
    const server = createPortfolioServer(config);
    server.on('error', error => { console.error(`Proxy: ${error.message}`); process.exitCode = 1; });
    server.listen(config.proxyPort, '127.0.0.1', () => console.log(`Portfolio API proxy ready on http://127.0.0.1:${config.proxyPort}`));
    for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => server.stop());
  } catch (error) { console.error(`Proxy: ${error.message}`); process.exitCode = 1; }
}

module.exports = { loadConfig, resolveRoute, createPortfolioServer, createDashboardLoader, DASHBOARD_KEYS };

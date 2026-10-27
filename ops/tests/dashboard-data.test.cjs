'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { PassThrough, Writable } = require('node:stream');
const { createDashboardLoader, createPortfolioServer, DASHBOARD_KEYS } = require('../portfolio-proxy.cjs');

const fixture = () => ({ ...Object.fromEntries(DASHBOARD_KEYS.map(key => [key, {}])),
  ALL_ALGOS: ['test-agent'], ALGO_MEANS: { 'test-agent': 123.456 }, SAVING_PCT: { 'test-agent': 7.89 } });
async function temp(t) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'hems-protocol-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  return dir;
}
test('serves unchanged JSON metrics with provenance, caches and refreshes edited exports', async t => {
  const filename = path.join(await temp(t), 'evaluation.json');
  const bytes = JSON.stringify(fixture()); await fs.writeFile(filename, bytes);
  const load = createDashboardLoader(filename);
  const first = await load();
  assert.deepEqual(first.data, fixture());
  assert.equal(first.source.sha256, createHash('sha256').update(bytes).digest('hex'));
  assert.equal(await load(), first);
  const changed = fixture(); changed.ALGO_MEANS['test-agent'] = 789.012;
  await fs.writeFile(filename, JSON.stringify(changed));
  assert.equal((await load()).data.ALGO_MEANS['test-agent'], 789.012);
  await fs.unlink(filename);
  await assert.rejects(load()); // No stale fallback if the host removes the export.
});
test('reads generated ESM exports without bundling research values in source', async t => {
  const filename = path.join(await temp(t), 'evaluation.mjs');
  await fs.writeFile(filename, Object.entries(fixture()).map(([key, value]) => `export const ${key} = ${JSON.stringify(value)};`).join('\n'));
  assert.deepEqual((await createDashboardLoader(filename)()).data, fixture());
});
test('rejects missing configuration, malformed/incomplete and fictional exports', async t => {
  await assert.rejects(createDashboardLoader()(), /not configured/);
  const filename = path.join(await temp(t), 'evaluation.json');
  for (const contents of ['{}', 'not JSON', '// FICTIONAL']) {
    await fs.writeFile(filename, contents);
    await assert.rejects(createDashboardLoader(filename)());
  }
  const bad = fixture(); bad.ALGO_MEANS['test-agent'] = null;
  await fs.writeFile(filename, JSON.stringify(bad));
  await assert.rejects(createDashboardLoader(filename)(), /Invalid evaluation metrics/);
});
class Response extends Writable {
  constructor() { super(); this.chunks = []; this.headersSent = false; }
  writeHead(status, headers) { this.status = status; this.headers = headers; this.headersSent = true; }
  _write(chunk, encoding, callback) { this.chunks.push(chunk); callback(); }
}
async function request(config, method = 'GET', origin = 'https://example.github.io') {
  const server = createPortfolioServer(config);
  const req = new PassThrough();
  Object.assign(req, { url: '/hems/dashboard-data', method, headers: { origin } });
  const res = new Response();
  const done = new Promise((resolve, reject) => { res.on('finish', () => resolve(res)); res.on('error', reject); });
  server.emit('request', req, res); req.end();
  return done;
}
test('gateway protects origin, allows only reads, supports HEAD and returns 503 without leaking local paths', async t => {
  const filename = path.join(await temp(t), 'evaluation.json');
  await fs.writeFile(filename, JSON.stringify(fixture()));
  const config = { allowedOrigins: ['https://example.github.io'], hemsDashboardDataFile: filename };
  const get = await request(config);
  assert.equal(get.status, 200);
  assert.equal(get.headers['access-control-allow-origin'], config.allowedOrigins[0]);
  assert.deepEqual(JSON.parse(Buffer.concat(get.chunks)).data, fixture());
  assert.equal((await request(config, 'POST')).status, 405);
  assert.equal((await request(config, 'GET', 'https://other.example')).status, 403);
  const head = await request(config, 'HEAD');
  assert.equal(head.status, 200); assert.equal(head.chunks.length, 0);
  await fs.unlink(filename);
  const missing = await request(config);
  assert.equal(missing.status, 503);
  assert.ok(!Buffer.concat(missing.chunks).toString().includes(filename));
});

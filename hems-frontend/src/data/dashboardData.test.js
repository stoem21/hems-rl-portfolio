import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DASHBOARD_KEYS, validateDashboardPayload, loadDashboardData } from './dashboardData.js';

// Small protocol fixtures only; no research results are bundled.
function payload() {
  const data = Object.fromEntries(DASHBOARD_KEYS.map(key => [key, {}]));
  for (const key of ['ALL_ALGOS', 'ERR_ALGOS', 'SIM_ALGOS']) data[key] = ['test-agent'];
  for (const key of ['ERROR_TYPES', 'ERROR_PCTS', 'SEEDS', 'HOUSEHOLDS', 'CLUSTERS', 'ALL_HOUSEHOLDS']) data[key] = [1];
  for (const key of ['DATE_DATASET_START', 'DATE_TRAIN_END', 'DATE_VAL_END', 'DATE_TEST_START']) data[key] = '2020-01-01';
  data.ALGO_MEANS = { 'test-agent': 123.456 };
  data.SAVING_PCT = { 'test-agent': 7.89 };
  return { schemaVersion: 1, source: { kind: 'saved-evaluation', sha256: 'a'.repeat(64), updatedAt: '2020-01-01T00:00:00Z' }, data };
}
test('loads backend values unchanged and passes cancellation without browser caching', async () => {
  const original = payload();
  const signal = new AbortController().signal;
  const result = await loadDashboardData(async (path, options) => {
    assert.equal(path, '/dashboard-data');
    assert.equal(options.signal, signal);
    assert.equal(options.cache, 'no-store');
    return { ok: true, json: async () => original };
  }, signal);
  assert.equal(result, original);
  assert.equal(result.data.ALGO_MEANS['test-agent'], 123.456);
});
test('fails when offline or HTTP fails, without substituting sample results', async () => {
  await assert.rejects(loadDashboardData(async () => { throw new Error('offline'); }), /offline/);
  await assert.rejects(loadDashboardData(async () => ({ ok: false, status: 503 })), /503/);
});
test('rejects missing exports, malformed metrics and untrusted response shape', () => {
  const missing = payload(); delete missing.data.SEED_HH_DATA;
  assert.throws(() => validateDashboardPayload(missing), /Missing evaluation field/);
  const bad = payload(); bad.data.ALGO_MEANS['test-agent'] = null;
  assert.throws(() => validateDashboardPayload(bad), /invalid evaluation metrics/);
  const old = payload(); old.source.kind = 'sample';
  assert.throws(() => validateDashboardPayload(old), /unsupported/);
  const date = payload(); date.data.DATE_TRAIN_END = 123;
  assert.throws(() => validateDashboardPayload(date), /evaluation date/);
});

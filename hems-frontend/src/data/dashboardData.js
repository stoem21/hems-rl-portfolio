// Data contract only. Evaluation values are fetched from the tunnel at runtime.
export const DASHBOARD_KEYS = ["ALGO_MEANS", "SAVING_PCT", "HH_DATA", "CLUSTER_DATA", "ROBUSTNESS", "NO_ERR_MEANS", "STRATEGY_COLORS", "STRATEGY_LABELS", "ALL_ALGOS", "ERR_ALGOS", "SIM_ALGOS", "ERROR_TYPES", "ERROR_PCTS", "DATE_DATASET_START", "DATE_TRAIN_END", "DATE_VAL_END", "DATE_TEST_START", "SEEDS", "HOUSEHOLDS", "CLUSTERS", "TRAIN_HOUSEHOLDS", "TEST_HOUSEHOLDS", "ALL_HOUSEHOLDS", "HH_META", "SEED_HH_DATA", "ROBUSTNESS_BY_HH"];

export function validateDashboardPayload(payload) {
  const data = payload?.data;
  if (payload?.schemaVersion !== 1 || payload?.source?.kind !== 'saved-evaluation' ||
      !/^[a-f0-9]{64}$/.test(payload?.source?.sha256 || '') ||
      !Number.isFinite(Date.parse(payload?.source?.updatedAt))) {
    throw new Error('The backend returned an unsupported evaluation response.');
  }
  for (const key of DASHBOARD_KEYS) {
    if (data?.[key] === undefined || data[key] === null) throw new Error(`Missing evaluation field: ${key}`);
  }
  for (const key of ['ALL_ALGOS', 'ERR_ALGOS', 'SIM_ALGOS', 'ERROR_TYPES', 'ERROR_PCTS', 'SEEDS', 'HOUSEHOLDS', 'CLUSTERS', 'ALL_HOUSEHOLDS']) {
    if (!Array.isArray(data[key]) || !data[key].length) throw new Error(`Invalid evaluation field: ${key}`);
  }
  for (const key of ['DATE_DATASET_START', 'DATE_TRAIN_END', 'DATE_VAL_END', 'DATE_TEST_START']) {
    if (typeof data[key] !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(data[key])) throw new Error(`Invalid evaluation date: ${key}`);
  }
  if (!data.ALL_ALGOS.every(key => Number.isFinite(data.ALGO_MEANS[key]) && Number.isFinite(data.SAVING_PCT[key]))) {
    throw new Error('The backend returned invalid evaluation metrics.');
  }
  return payload;
}

export async function loadDashboardData(fetchApi, signal) {
  const response = await fetchApi('/dashboard-data', { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Evaluation backend unavailable (HTTP ${response.status}).`);
  return validateDashboardPayload(await response.json());
}

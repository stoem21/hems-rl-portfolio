export function apiError(data) {
  const detail = data?.detail ?? data?.error;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((e) => `${(e.loc || []).slice(1).join(" ")}: ${e.msg}`).join("; ");
  return detail?.message || "The request could not be completed.";
}

export function simulationKey(r) {
  return `${r.agent}-${r.seed}-${r.error_type}-${r.error_pct}`;
}

export function manualSimulationRequest(settings, agents = null) {
  const selected = agents ?? [settings.agent];
  if (agents !== null && (agents.length < 2 || agents.length > 12 || new Set(agents).size !== agents.length)) {
    throw new Error("Select 2–12 different agents to compare.");
  }
  const { agent, ...shared } = settings;
  const hasForecast = selected.some((name) => !name.startsWith("R-"));
  const body = {
    ...shared,
    error_type: hasForecast ? settings.error_type : "none",
    error_pct: hasForecast && settings.error_type !== "none" ? settings.error_pct : 0,
  };
  return agents === null
    ? { path: "/simulate", body: { ...body, agent } }
    : { path: "/simulate/compare", body: { ...body, agents: [...agents] } };
}

export function mergeSimulationResults(previous, incoming) {
  if (!incoming.length) return previous;
  const sameWindow = (r) => ["household_id", "start_date", "n_days", "seed"].every((key) => r[key] === incoming[0][key]);
  const results = previous.filter(sameWindow);
  for (const result of incoming) {
    const position = results.findIndex((r) => simulationKey(r) === simulationKey(result));
    if (position < 0) results.push(result);
    else results[position] = result;
  }
  return results;
}

export async function readAskEvents(response, onEvent) {
  if (!response.ok) throw new Error(apiError(await response.json()));
  if (!response.body) throw new Error("Streaming is unavailable. Please retry.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  function frame(text) {
    const data = text.split(/\r?\n/).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
    if (!data) return;
    const event = JSON.parse(data);
    if (event.type === "done") completed = true;
    onEvent(event);
  }
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        frame(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary[0].length);
      }
      if (done) break;
    }
    if (buffer.trim()) frame(buffer);
    if (!completed) throw new Error("The connection ended before the answer finished. Completed simulation results remain available.");
  } finally {
    reader.releaseLock();
  }
}

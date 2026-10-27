import test from "node:test";
import assert from "node:assert/strict";
import { readAskEvents, mergeSimulationResults, apiError, manualSimulationRequest } from "./simulationStream.js";

function response(parts) {
  return new Response(new ReadableStream({ start(controller) {
    for (const bytes of parts) controller.enqueue(bytes);
    controller.close();
  } }));
}

test("SSE handles network fragmentation, UTF-8 splits and CRLF across chunks", async () => {
  const encoder = new TextEncoder();
  const text = ': keepalive\r\ndata: {"type":"text","text":"สวัสดี"}\r\n\r\ndata: {"type":"done"}\r\n\r\n';
  const events = [];
  await readAskEvents(response(Array.from(encoder.encode(text), (byte) => new Uint8Array([byte]))), (e) => events.push(e));
  assert.deepEqual(events, [{ type: "text", text: "สวัสดี" }, { type: "done" }]);
});

test("an interrupted response preserves delivered results but reports failure", async () => {
  const events = [];
  await assert.rejects(readAskEvents(response([new TextEncoder().encode('data: {"type":"simulation","results":[{"agent":"I-TD3"}]}\n\n')]), (e) => events.push(e)), /connection ended/);
  assert.equal(events[0].results[0].agent, "I-TD3");
});

test("comparison accumulation replaces duplicates and never mixes date windows", () => {
  const r = { agent: "I-TD3", seed: 1, household_id: 1, start_date: "2012-08-01", n_days: 1, error_type: "none", error_pct: 0 };
  const a = mergeSimulationResults([r], [{ ...r, agent: "R-TD3" }]);
  assert.equal(a.length, 2);
  assert.equal(mergeSimulationResults(a, [r]).length, 2);
  const b = mergeSimulationResults(a, [{ ...r, start_date: "2012-09-01" }]);
  assert.equal(b.length, 1);
  assert.equal(b[0].start_date, "2012-09-01");
});

test("data gaps and input validation become readable error messages", () => {
  assert.equal(apiError({ detail: { code: "data_gap", message: "Missing 48 intervals" } }), "Missing 48 intervals");
  assert.match(apiError({ detail: [{ loc: ["body", "seed"], msg: "must be <= 5" }] }), /seed: must be <= 5/);
});

const settings = { household_id: 1, start_date: "2012-08-01", n_days: 1, seed: 1, agent: "R-TD3", error_type: "noise", error_pct: 20 };

test("manual comparison sends chosen agents and common settings without a forbidden single-agent field", () => {
  const request = manualSimulationRequest(settings, ["R-TD3", "I-TD3", "N-SAC"]);
  assert.equal(request.path, "/simulate/compare");
  assert.deepEqual(request.body.agents, ["R-TD3", "I-TD3", "N-SAC"]);
  assert.equal("agent" in request.body, false);
  assert.equal(request.body.error_type, "noise", "A reactive first agent must not disable errors for forecast agents");
  assert.equal(request.body.error_pct, 20);
  assert.equal(request.body.start_date, settings.start_date);
});

test("manual comparison refuses insufficient, duplicated and excessive choices", () => {
  for (const agents of [[], ["I-TD3"], ["I-TD3", "I-TD3"], Array.from({ length: 13 }, (_, i) => String(i))]) {
    assert.throws(() => manualSimulationRequest(settings, agents), /2–12/);
  }
});

test("single and all-reactive manual runs normalize unused error settings", () => {
  const single = manualSimulationRequest(settings);
  assert.equal(single.path, "/simulate");
  assert.equal(single.body.agent, "R-TD3");
  assert.equal(single.body.error_type, "none");
  assert.equal(single.body.error_pct, 0);
  const reactive = manualSimulationRequest(settings, ["R-TD3", "R-SAC"]);
  assert.equal(reactive.body.error_type, "none");
  assert.equal(reactive.body.error_pct, 0);
});

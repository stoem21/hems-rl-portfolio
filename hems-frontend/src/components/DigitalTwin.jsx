import { useState, useEffect, useMemo, useRef } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  Cell,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SimulationComparison from "./SimulationComparison";
import { apiError, readAskEvents, mergeSimulationResults, manualSimulationRequest } from "../lib/simulationStream.js";
import { apiFetch } from "../config";
import { useBackendHealth } from "../hooks/useBackendHealth";
import { useDashboardData } from '../data/DashboardDataContext.jsx';

const flat = (obj) => Object.values(obj).flat();
const UVICORN_CMD =
  "cd hems-backend && uvicorn main:app --host 0.0.0.0 --port 8000";

// ── Time-of-use classification (NSW Ausgrid style) by hour of day ──
const TOU_FILL = { peak: "#D85A30", shoulder: "#E3B23C", offpeak: "#888780" };
const TOU_OPACITY = { peak: 0.08, shoulder: 0.05, offpeak: 0.04 };
function buildTouAreas(rates) {
  const areas = [];
  rates.forEach((rate, i) => {
    const period = rate >= 0.45 ? "peak" : rate <= 0.27 ? "offpeak" : "shoulder";
    if (areas.length && areas.at(-1).period === period) areas.at(-1).x2 = i + 1;
    else areas.push({ x1: i, x2: i + 1, period });
  });
  return areas;
}

export default function DigitalTwin() {
  const {
    SEEDS,
    TRAIN_HOUSEHOLDS,
    TEST_HOUSEHOLDS,
    HH_META,
    SIM_ALGOS,
    ERROR_TYPES,
    DATE_DATASET_START,
    DATE_TRAIN_END,
    DATE_VAL_END,
  } = useDashboardData();
  const TEST_FLAT = flat(TEST_HOUSEHOLDS);
  const TRAIN_FLAT = flat(TRAIN_HOUSEHOLDS);
  const OTHER_FLAT = Array.from({ length: 300 }, (_, i) => i + 1).filter(
    (id) => !TEST_FLAT.includes(id) && !TRAIN_FLAT.includes(id),
  );

  function periodBadge(dateStr) {
    if (!dateStr) return null;
    if (dateStr < DATE_TRAIN_END)
      return {
        cls: "amber",
        text: "⚠ Training period — agent has seen this data, results may be optimistic",
      };
    if (dateStr < DATE_VAL_END)
      return {
        cls: "amber",
        text: "⚠ Validation period — agent was tuned on this data",
      };
    return {
      cls: "green",
      text: "✓ Test period — held-out data, results are unbiased",
    };
  }

  function householdBadge(id) {
    const role = HH_META[id]?.role;
    if (role === "train")
      return {
        cls: "blue",
        text: "Training household — agent was trained on households from this cluster",
      };
    if (role === "unused")
      return { cls: "gray", text: "Unseen household — not used in any phase" };
    return null; // test households: expected use case, no badge
  }


  const online = useBackendHealth();

  const [household, setHousehold] = useState(1);
  const [minDate, setMinDate] = useState(DATE_DATASET_START);
  const [maxDate, setMaxDate] = useState("");
  const [startDate, setStartDate] = useState(DATE_VAL_END); // 2012-07-01 test start
  const [nDays, setNDays] = useState(1);
  const [agent, setAgent] = useState("I-TD3");
  const [runMode, setRunMode] = useState("single");
  const [compareAgents, setCompareAgents] = useState(["I-TD3", "R-TD3"]);
  const [seed, setSeed] = useState(1);
  const [errorType, setErrorType] = useState("none");
  const [errorPct, setErrorPct] = useState(10);

  const comparing = runMode === "compare";
  const selectionValid = !comparing || compareAgents.length >= 2;
  const validationAgent = comparing ? compareAgents[0] || agent : agent;
  const isNoForecast = comparing ? compareAgents.length > 0 && compareAgents.every((name) => name.startsWith("R-")) : agent.startsWith("R-");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [explainSource, setExplainSource] = useState("");
  const [explaining, setExplaining] = useState(false);
  const [engine, setEngine] = useState("");
  const [engines, setEngines] = useState([]);
  const [question, setQuestion] = useState("");
  const [askAnswer, setAskAnswer] = useState("");
  const [askSource, setAskSource] = useState("");
  const [asking, setAsking] = useState(false);
  // Independent from `engine` above — Explanation and Ask can each use a
  // different model (e.g. Claude for the auto-explanation, local for follow-ups).
  const [askEngine, setAskEngine] = useState("");
  const [askSteps, setAskSteps] = useState([]);
  const [askError, setAskError] = useState("");
  const [toolResults, setToolResults] = useState([]);
  const [resultsOrigin, setResultsOrigin] = useState("ask");
  const [windowCheck, setWindowCheck] = useState({ loading: true, valid: false, message: "Checking data availability…" });
  const askController = useRef(null);
  const explainController = useRef(null);
  const resultsRef = useRef(null);
  useEffect(() => () => {
    askController.current?.abort();
    explainController.current?.abort();
  }, []);

  useEffect(() => {
    if (!online) return;
    const controller = new AbortController();
    setWindowCheck({ loading: true, valid: false, message: "Checking data availability…" });
    const timer = setTimeout(async () => {
      try {
        const response = await apiFetch(`/simulate/validate`, {
          method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
          body: JSON.stringify({ household_id: household, start_date: startDate, n_days: nDays, agent: validationAgent, seed,
            error_type: isNoForecast ? "none" : errorType, error_pct: isNoForecast || errorType === "none" ? 0 : errorPct }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(apiError(data));
        if (!controller.signal.aborted) setWindowCheck({ loading: false, valid: true, message: data.message });
      } catch (err) {
        if (!controller.signal.aborted) setWindowCheck({ loading: false, valid: false, message: err.message });
      }
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [online, household, startDate, nDays, validationAgent, seed, errorType, errorPct, isNoForecast]);

  // Fetch the available date range whenever the household changes.
  useEffect(() => {
    if (!online) return;
    let cancelled = false;
    apiFetch(`/household-range?id=${household}`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled && d.max_date) { setMaxDate(d.max_date); setMinDate(d.min_date); }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [household, online]);

  // Load the available explanation engines once; default to the local model.
  useEffect(() => {
    if (!online) return;
    apiFetch(`/models`)
      .then((r) => r.json())
      .then((d) => {
        setEngines(d.engines || []);
        setEngine((cur) => cur || d.default || "");
        setAskEngine((cur) => cur || d.default || "");
      })
      .catch(() => {});
  }, [online]);

  const hhBadge = householdBadge(household);
  const dtBadge = periodBadge(startDate);

  const chartData = useMemo(() => {
    if (!result) return [];
    return result.soc_timeline.map((soc, i) => ({
      i,
      soc,
      dispatch: result.dispatch_timeline[i],
      load: result.load_timeline[i],
      pv: result.pv_timeline[i],
      price: result.tou_rates[i],
    }));
  }, [result]);

  const touAreas = useMemo(
    () => (result ? buildTouAreas(result.tou_rates) : []),
    [result],
  );

  async function runSimulation(e) {
    e.preventDefault();
    if (!windowCheck.valid || windowCheck.loading || asking || loading || !selectionValid) return;
    explainController.current?.abort();
    setToolResults([]);
    setAskSteps([]);
    setAskError("");
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation("");
    setExplainSource("");
    setExplaining(false);
    setQuestion("");
    setAskAnswer("");
    setAskSource("");
    setAsking(false);
    try {
      const request = manualSimulationRequest({
        household_id: household, start_date: startDate, n_days: nDays,
        agent, seed, error_type: errorType, error_pct: errorPct,
      }, comparing ? compareAgents : null);
      const r = await apiFetch(request.path, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request.body),
      });
      const data = await r.json();
      if (!r.ok || data.error) {
        setError(apiError(data));
        return;
      }
      setResultsOrigin("manual");
      if (comparing) {
        setToolResults(data.results);
        selectSimulation(data.results[0], true);
      } else {
        setResult(data);
        streamExplanation(data);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function streamExplanation(data, engineId = engine) {
    const sel = engines.find((e) => e.id === engineId) || {};
    explainController.current?.abort();
    const controller = new AbortController();
    explainController.current = controller;
    setExplanation("");
    setExplaining(true);
    try {
      const r = await apiFetch(`/explain`, {
        signal: controller.signal,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          household_id: data.household_id,
          household_role: data.household_role,
          period: data.period,
          agent: data.agent,
          seed: data.seed,
          annual_cost: data.annual_cost,
          saving_pct: data.saving_pct,
          error_type: data.error_type,
          error_pct: data.error_pct,
          n_days: data.n_days,
          start_date: data.start_date,
          ...(data.habit || {}),
          engine: sel.engine || "",
          model: sel.model || "",
        }),
      });
      if (!r.ok) throw new Error(apiError(await r.json()));
      setExplainSource(r.headers.get("X-Explain-Source") || "");
      if (!r.body) {
        setExplanation(await r.text());
        return;
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setExplanation(acc);
      }
    } catch (err) {
      if (!controller.signal.aborted) setExplanation(`Explanation unavailable: ${err.message}`);
    } finally {
      if (explainController.current === controller) setExplaining(false);
    }
  }

  function selectSimulation(data, preserveForecast = false) {
    explainController.current?.abort();
    setExplaining(false);
    setExplanation("");
    setExplainSource("");
    setError(null);
    setResult(data);
    setHousehold(data.household_id);
    setStartDate(data.start_date);
    setNDays(data.n_days);
    setAgent(data.agent);
    setSeed(data.seed);
    if (!preserveForecast) {
      setErrorType(data.error_type);
      setErrorPct(data.error_pct);
    }
  }

  async function askQuestion(e) {
    e.preventDefault();
    if (!question.trim() || asking || loading) return;
    explainController.current?.abort();
    setExplaining(false);
    askController.current?.abort();
    const controller = new AbortController();
    askController.current = controller;
    const sel = engines.find((en) => en.id === askEngine) || {};
    setAsking(true);
    setAskAnswer("");
    setAskSource(sel.label || "AI assistant");
    setAskSteps([]);
    setAskError("");
    setToolResults([]);
    setResultsOrigin("ask");
    let answer = "";
    try {
      const r = await apiFetch(`/ask`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({
          question, stream_events: true,
          ...(result ? {
            household_id: result.household_id, household_role: result.household_role,
            period: result.period, agent: result.agent, seed: result.seed,
            annual_cost: result.annual_cost, saving_pct: result.saving_pct,
            error_type: result.error_type, error_pct: result.error_pct,
            n_days: result.n_days, start_date: result.start_date, ...(result.habit || {}),
          } : {}),
          engine: sel.engine || "", model: sel.model || "",
        }),
      });
      await readAskEvents(r, (event) => {
        if (controller.signal.aborted) return;
        if (event.type === "text") { answer += event.text; setAskAnswer(answer); }
        else if (event.type === "simulation" && event.results?.length) {
          setToolResults((previous) => mergeSimulationResults(previous, event.results));
          selectSimulation(event.results[0]);
        } else if (event.type === "status") {
          const id = event.tool_id || event.stage;
          setAskSteps((previous) => {
            const next = previous.map((step) => step.state === "running" && !step.tool_id && step.id !== id ? { ...step, state: "complete" } : step);
            const position = next.findIndex((step) => step.id === id);
            const item = { ...event, id, state: event.state || "running" };
            if (position < 0) next.push(item); else next[position] = item;
            return next;
          });
        } else if (event.type === "error") {
          setAskError(event.message);
          setAskSteps((steps) => steps.map((step) => step.state === "running" ? { ...step, state: "failed" } : step));
        } else if (event.type === "done") {
          setAskSteps((steps) => steps.map((step) => step.state === "running" ? { ...step, state: "complete" } : step));
        }
      });
    } catch (err) {
      if (!controller.signal.aborted) {
        setAskError(err.message);
        setAskSteps((steps) => steps.map((step) => step.state === "running" ? { ...step, state: "failed" } : step));
      }
    } finally {
      if (askController.current === controller) setAsking(false);
    }
  }

  if (!online) {
    return (
      <div className="page">
        <h2>
          Simulation <span className="status-offline">● Backend offline</span>
        </h2>
        <div className="card">
          <p>
            The simulation backend is not reachable. Start it locally, then this
            page will reconnect automatically.
          </p>
          <div className="code-block">
            <code>{UVICORN_CMD}</code>
            <button
              className="ghost"
              onClick={() => navigator.clipboard?.writeText(UVICORN_CMD)}
            >
              Copy
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <h2>
        Simulation{" "}
        <span className="status-online" style={{ fontSize: "0.85rem" }}>
          ● Backend online
        </span>
      </h2>
      <p className="subtitle">
        Run a live battery-dispatch simulation against a trained agent
        checkpoint.
      </p>

      <form className="card" onSubmit={runSimulation}>
        <fieldset className="run-mode" disabled={loading || asking}>
          <legend>Run mode</legend><div className="radio-row">
            <label><input type="radio" name="run-mode" value="single" checked={!comparing} onChange={() => setRunMode("single")}/>Single agent</label>
            <label><input type="radio" name="run-mode" value="compare" checked={comparing} onChange={() => setRunMode("compare")}/>Compare agents</label>
          </div>
        </fieldset>
        <div className="controls">
          <div className="field">
            <label htmlFor="hhsel">Household</label>
            <select
              id="hhsel"
              value={household}
              onChange={(e) => setHousehold(Number(e.target.value))}
            >
              <optgroup label={`Test households (${TEST_FLAT.length})`}>
                {TEST_FLAT.map((id) => (
                  <option key={id} value={id}>
                    HH {id} · Cluster {HH_META[id]?.cluster ?? "–"} · TEST
                  </option>
                ))}
              </optgroup>
              <optgroup label={`Train households (${TRAIN_FLAT.length})`}>
                {TRAIN_FLAT.map((id) => (
                  <option key={id} value={id}>
                    HH {id} · Cluster {HH_META[id]?.cluster ?? "–"} · TRAIN
                  </option>
                ))}
              </optgroup>
              <optgroup label={`Other households (${OTHER_FLAT.length})`}>
                {OTHER_FLAT.map((id) => (
                  <option key={id} value={id}>
                    HH {id} · unused
                  </option>
                ))}
              </optgroup>
            </select>
          </div>

          <div className="field">
            <label htmlFor="date">Start date</label>
            <input
              id="date"
              type="date"
              value={startDate}
              min={minDate}
              max={maxDate || undefined}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div className="field">
            <label htmlFor="ndays">Number of days: {nDays}</label>
            <input
              id="ndays"
              type="range"
              min={1}
              max={30}
              value={nDays}
              onChange={(e) => setNDays(Number(e.target.value))}
            />
          </div>

          {comparing ? <fieldset className="comparison-agent-picker" disabled={loading || asking}>
            <legend>Agents to compare ({compareAgents.length}/12 selected)</legend>
            <p className="comparison-note">Choose 2–12 agents. All use the same household, date window and seed. No AI request is needed.</p>
            <div className="agent-checkboxes">{SIM_ALGOS.map((name) => <label key={name} className={compareAgents.includes(name) ? "chosen" : ""}>
              <input type="checkbox" checked={compareAgents.includes(name)} onChange={(e) => setCompareAgents((previous) => e.target.checked ? [...previous, name] : previous.filter((item) => item !== name))}/>{name}
            </label>)}</div>
            <div className="selection-actions"><button className="ghost" type="button" onClick={() => setCompareAgents([...SIM_ALGOS])}>Select all</button><button className="ghost" type="button" onClick={() => setCompareAgents([])}>Clear selection</button></div>
            {!selectionValid && <p role="alert" className="selection-error">Select at least 2 agents to compare.</p>}
            {compareAgents.some((name) => name.startsWith("R-")) && !isNoForecast && <p className="comparison-note">Forecast errors apply to forecast-based agents. Reactive agents ignore them.</p>}
          </fieldset> : <div className="field">
            <label htmlFor="agentsel">Agent</label>
            <select id="agentsel" value={agent} onChange={(e) => setAgent(e.target.value)}>
              {SIM_ALGOS.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </div>}


          <div className="field">
            <label>Seed (run)</label>
            <div className="radio-row">
              {SEEDS.map((s) => (
                <label key={s}>
                  <input
                    type="radio"
                    name="seed"
                    value={s}
                    checked={seed === s}
                    onChange={() => setSeed(s)}
                  />
                  {s}
                </label>
              ))}
            </div>
          </div>

          <div className="field" style={isNoForecast ? { opacity: 0.4, pointerEvents: "none" } : undefined}>
            <label htmlFor="errtype">
              Forecast error type{isNoForecast && " (not used by selected agents)"}
            </label>
            <select
              id="errtype"
              value={isNoForecast ? "none" : errorType}
              disabled={isNoForecast}
              onChange={(e) => setErrorType(e.target.value)}
            >
              <option value="none">none</option>
              {ERROR_TYPES.map((et) => (
                <option key={et} value={et}>
                  {et}
                </option>
              ))}
            </select>
          </div>

          <div className="field" style={isNoForecast ? { opacity: 0.4, pointerEvents: "none" } : undefined}>
            <label htmlFor="errpct">
              Severity: {isNoForecast || errorType === "none" ? "—" : `${errorPct}%`}
            </label>
            <input
              id="errpct"
              type="range"
              min={0}
              max={20}
              value={errorPct}
              disabled={isNoForecast || errorType === "none"}
              onChange={(e) => setErrorPct(Number(e.target.value))}
            />
          </div>
        </div>

        {(hhBadge || dtBadge) && (
          <div style={{ marginTop: 16 }}>
            {hhBadge && (
              <span className={`badge ${hhBadge.cls}`}>{hhBadge.text}</span>
            )}
            {dtBadge && (
              <span className={`badge ${dtBadge.cls}`}>{dtBadge.text}</span>
            )}
          </div>
        )}

        <div role="status" className={`window-check ${windowCheck.valid ? "valid" : windowCheck.loading ? "pending" : "invalid"}`}>{windowCheck.message}</div>
        <div style={{ marginTop: 16 }}>
          <button className="primary" type="submit" disabled={loading || asking || windowCheck.loading || !windowCheck.valid || !selectionValid}>
            {loading ? (
              <>
                <span className="spinner" /> {comparing ? `Comparing ${compareAgents.length} agents…` : "Simulating…"}
              </>
            ) : (
              comparing ? "Run comparison" : "Run simulation"
            )}
          </button>
        </div>
      </form>

      {error && <div role="alert" className="badge red">{error}</div>}
      <div ref={resultsRef}><SimulationComparison results={toolResults} selected={result} onSelect={(data) => selectSimulation(data, resultsOrigin === "manual" && comparing)}/></div>

      {result && (
        <>
          <div className="card">
            <div className="result-grid">
              <div className="result-stat">
                <div className="k">Period cost</div>
                <div className="v">AUD {result.period_cost.toFixed(2)}</div>
              </div>
              <div className="result-stat">
                <div className="k">Annualised cost</div>
                <div className="v">AUD {result.annual_cost.toFixed(0)}</div>
              </div>
              <div className="result-stat">
                <div className="k">No-battery (annual)</div>
                <div className="v">AUD {result.baseline_annual.toFixed(0)}</div>
              </div>
              <div className="result-stat">
                <div className="k">Saving vs no battery</div>
                <div className="v">{result.saving_pct.toFixed(1)}%</div>
              </div>
            </div>
            <div className="period-label">
              Simulating HH {result.household_id} · {result.n_days} day(s) from{" "}
              {result.start_date} · {result.period} period · {result.agent} seed{" "}
              {result.seed}
              {result.error_type !== "none" &&
                ` · ${result.error_type} @ ${result.error_pct}%`}
            </div>

            {result.warnings?.map((warning) => <p key={warning} className="badge amber">{warning}</p>)}
            <h3>Battery dispatch &amp; state of charge</h3>
            <ResponsiveContainer width="100%" height={380}>
              <ComposedChart
                data={chartData}
                margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(0,0,0,0.06)"
                />
                {touAreas.map((a, k) => (
                  <ReferenceArea
                    key={k}
                    x1={a.x1}
                    x2={a.x2}
                    yAxisId="left"
                    fill={TOU_FILL[a.period]}
                    fillOpacity={TOU_OPACITY[a.period]}
                    stroke="none"
                  />
                ))}
                {result.n_days > 1 &&
                  Array.from({ length: result.n_days - 1 }, (_, k) => (
                    <ReferenceLine
                      key={`d${k}`}
                      x={(k + 1) * 48}
                      yAxisId="left"
                      stroke="rgba(0,0,0,0.25)"
                      strokeDasharray="4 3"
                      label={{
                        value: `Day ${k + 2}`,
                        fontSize: 10,
                        fill: "#73726c",
                      }}
                    />
                  ))}
                <XAxis
                  dataKey="i"
                  tick={{ fontSize: 11, fill: "#73726c" }}
                  label={{
                    value: "30-min interval",
                    position: "insideBottom",
                    offset: -2,
                    fontSize: 11,
                    fill: "#73726c",
                  }}
                />
                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 11, fill: "#73726c" }}
                  label={{
                    value: "Dispatch kW",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                    fill: "#73726c",
                  }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: "#73726c" }}
                  label={{
                    value: "SoC",
                    angle: 90,
                    position: "insideRight",
                    fontSize: 11,
                    fill: "#73726c",
                  }}
                />
                <Tooltip />
                <Bar yAxisId="left" dataKey="dispatch" name="Dispatch kW">
                  {chartData.map((d, k) => (
                    <Cell
                      key={k}
                      fill={d.dispatch >= 0 ? "#1D9E75" : "#D85A30"}
                    />
                  ))}
                </Bar>
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="soc"
                  name="SoC"
                  stroke="#378ADD"
                  strokeWidth={2}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h3>Load vs PV generation</h3>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart
                data={chartData}
                margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(0,0,0,0.06)"
                />
                <XAxis dataKey="i" tick={{ fontSize: 11, fill: "#73726c" }} />
                <YAxis
                  tick={{ fontSize: 11, fill: "#73726c" }}
                  label={{
                    value: "kW",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                    fill: "#73726c",
                  }}
                />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="load" name="Load" fill="#888780" />
                <Bar dataKey="pv" name="PV" fill="#E3B23C" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <div className="explain-head">
              <h3>
                Explanation
                {explainSource && (
                  <span className="explain-source">{explainSource}</span>
                )}
              </h3>
              {engines.length > 0 && (
                <label className="explain-engine">
                  Model:{" "}
                  <select
                    value={engine}
                    disabled={explaining}
                    onChange={(e) => {
                      const v = e.target.value;
                      setEngine(v);
                      if (result) streamExplanation(result, v);
                    }}
                  >
                    {engines.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            {!explaining && !explanation && <button type="button" className="ghost" onClick={() => streamExplanation(result)}>Generate explanation</button>}
            {explaining && !explanation ? (
              <div className="explain-loading">
                <span className="spinner" />
                <span>
                  Generating explanation… the local model is still computing
                  (this can take a while on CPU).
                </span>
              </div>
            ) : (
              <div className="explain-md">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {explanation || (resultsOrigin === 'manual' ? 'The simulation results are ready. Generate an explanation if you want an AI summary of this selected agent.' : 'This result was requested through Ask. See the answer below, or generate a separate explanation.')}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </>
      )}

      <div className="card">
        <div className="explain-head">
          <h3>
            Ask the agent
            {askSource && <span className="explain-source">{askSource}</span>}
          </h3>
          {engines.length > 0 && (
            <label className="explain-engine">
              Model:{" "}
              <select
                value={askEngine}
                disabled={asking}
                onChange={(e) => setAskEngine(e.target.value)}
              >
                {engines.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem", margin: "0 0 12px" }}>
          {result
            ? "Ask about this simulation, or anything else about the project — the agent can search the thesis or run a new comparison."
            : "Run a simulation for context-aware answers, or ask a general question about the project now — the agent can search the thesis or run its own simulation."}
        </p>
        <form
          onSubmit={askQuestion}
          style={{ display: "flex", gap: 8, marginBottom: 12 }}
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            aria-label="Ask a question"
            placeholder="e.g. Compare I-TD3 and R-TD3 for HH 1, 2012-08-01, 7 days"
            disabled={asking}
            style={{ flex: 1 }}
          />
          <button
            className="primary"
            type="submit"
            disabled={asking || loading || !question.trim()}
          >
            Ask
          </button>
        </form>
        {askSteps.length > 0 && <ol className="ask-progress" aria-label="Request progress" aria-live="polite">
          {askSteps.map((step) => <li key={step.id} className={step.state}>
            {step.state === "running" ? <span className="spinner"/> : <span aria-hidden="true">{step.state === "failed" ? "!" : "✓"}</span>}
            <span>{step.label}</span><span className="progress-state">{step.state}</span>
          </li>)}
        </ol>}
        {toolResults.length > 0 && <p className="charts-updated" role="status">Charts updated with {toolResults.length} calculated simulation(s). <button type="button" className="ghost" onClick={() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}>View results</button></p>}
        {askError && <p className="ask-error" role="alert">{askError}</p>}
        {asking && !askAnswer && <p className="comparison-note">The local model is working. Progress and calculated results appear as each step finishes.</p>}
        {askAnswer && <div className="explain-md"><p className="answer-label">AI explanation</p><ReactMarkdown remarkPlugins={[remarkGfm]}>{askAnswer}</ReactMarkdown></div>}

      </div>
    </div>
  );
}

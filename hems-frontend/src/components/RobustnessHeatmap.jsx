import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from "recharts";
import { useDashboardData } from '../data/DashboardDataContext.jsx';

const ERROR_LABELS = {
  noise: "Noise",
  drift_p: "Drift +",
  drift_n: "Drift −",
  bias_p: "Bias +",
  bias_n: "Bias −",
};

// Map a degradation fraction to a green→yellow→red background colour.
function degradeColor(pctChange) {
  // pctChange: positive = worse (higher cost). Clamp to [-2, +6]% for the scale.
  const t = Math.max(0, Math.min(1, (pctChange + 2) / 8));
  const hue = (1 - t) * 130; // 130=green -> 0=red
  return `hsl(${hue}, 62%, 72%)`;
}

export default function RobustnessHeatmap() {
  const {
    ROBUSTNESS,
    ROBUSTNESS_BY_HH,
    NO_ERR_MEANS,
    SEED_HH_DATA,
    SEEDS,
    ERR_ALGOS,
    ERROR_TYPES,
    ERROR_PCTS,
    HOUSEHOLDS,
    STRATEGY_COLORS,
  } = useDashboardData();
  const [agent, setAgent] = useState("I-TD3");
  const [lineError, setLineError] = useState("bias_n");
  const [house, setHouse] = useState("all"); // 'all' = mean over households, else household id (string)

  const isAll = house === "all";

  // Cost for (agent, error_type, pct), from aggregated means or a single household.
  const costAt = (a, et, p) =>
    isAll ? ROBUSTNESS[a][et][p] : ROBUSTNESS_BY_HH[house][et][p][a];

  // Clean (no-error) baseline for an agent — aggregated mean, or this household's
  // seed-averaged no-error cost.
  const cleanFor = (a) => {
    if (isAll) return NO_ERR_MEANS[a];
    const vals = SEEDS.map((s) => SEED_HH_DATA[s][house][a]);
    return vals.reduce((x, y) => x + y, 0) / vals.length;
  };

  const base = cleanFor(agent);

  // Line chart: cost vs severity for all 9 agents on the selected error type.
  const lineData = ERROR_PCTS.map((p) => {
    const row = { severity: `${parseFloat(p)}%` };
    ERR_ALGOS.forEach((a) => {
      row[a] = costAt(a, lineError, p);
    });
    return row;
  });

  const scope = isAll ? "all households (mean)" : `household ${house}`;

  return (
    <div className="page">
      <h2>Robustness heatmap</h2>
      {/* <p className="subtitle">Mean cost under injected forecast error. Greener = more robust, redder = more degradation.</p> */}

      <div className="card">
        <div
          style={{
            display: "flex",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 18,
          }}
        >
          <div className="field" style={{ minWidth: 160 }}>
            <label htmlFor="agent">Agent</label>
            <select
              id="agent"
              value={agent}
              onChange={(e) => setAgent(e.target.value)}
            >
              {ERR_ALGOS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ minWidth: 200 }}>
            <label htmlFor="house">Household</label>
            <select
              id="house"
              value={house}
              onChange={(e) => setHouse(e.target.value)}
            >
              <option value="all">All households (mean)</option>
              {HOUSEHOLDS.map((h) => (
                <option key={h} value={String(h)}>
                  HH {h}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="heatmap">
          <table>
            <thead>
              <tr>
                <th></th>
                {ERROR_PCTS.map((p) => (
                  <th key={p}>{parseFloat(p)}%</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ERROR_TYPES.map((et) => (
                <tr key={et}>
                  <td className="rowhead">{ERROR_LABELS[et]}</td>
                  {ERROR_PCTS.map((p) => {
                    const cost = costAt(agent, et, p);
                    const change = ((cost - base) / base) * 100;
                    return (
                      <td
                        key={p}
                        style={{ background: degradeColor(change) }}
                        title={`${change >= 0 ? "+" : ""}${change.toFixed(2)}% vs clean`}
                      >
                        {cost.toFixed(0)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="subtitle" style={{ margin: "12px 0 0" }}>
          Clean (no error) for {agent} · {scope}: AUD {base.toFixed(0)}
        </p>
      </div>

      <div className="card">
        <div className="field" style={{ maxWidth: 260, marginBottom: 16 }}>
          <label htmlFor="lineErr">Compare all agents on error type</label>
          <select
            id="lineErr"
            value={lineError}
            onChange={(e) => setLineError(e.target.value)}
          >
            {ERROR_TYPES.map((et) => (
              <option key={et} value={et}>
                {ERROR_LABELS[et]}
              </option>
            ))}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={lineData}
            margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
            <XAxis
              dataKey="severity"
              tick={{ fontSize: 12, fill: "#73726c" }}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fontSize: 12, fill: "#73726c" }}
              label={{
                value: "AUD / year",
                angle: -90,
                position: "insideLeft",
                fill: "#73726c",
                fontSize: 12,
              }}
            />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {/* Clean (no-error) baseline for the selected agent. */}
            <ReferenceLine
              y={base}
              stroke={STRATEGY_COLORS[agent]}
              strokeDasharray="6 4"
              label={{
                value: `${agent} clean`,
                position: "insideTopLeft",
                fill: STRATEGY_COLORS[agent],
                fontSize: 11,
              }}
            />
            {ERR_ALGOS.map((a) => (
              <Line
                key={a}
                type="monotone"
                dataKey={a}
                stroke={STRATEGY_COLORS[a]}
                strokeWidth={a === agent ? 3 : 1.4}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

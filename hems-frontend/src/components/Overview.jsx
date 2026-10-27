import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from "recharts";
import { useDashboardData } from '../data/DashboardDataContext.jsx';

const GROUPS = [
  { color: "#D85A30", name: "Baseline (no battery / rule-based)" },
  { color: "#888780", name: "Reactive (R)" },
  { color: "#378ADD", name: "Vanilla (V)" },
  { color: "#1D9E75", name: "Noise-trained (N)" },
  { color: "#639922", name: "Interval (I)" },
];

function CostTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="card" style={{ margin: 0, padding: "10px 14px" }}>
      <strong>{d.agent}</strong>
      <div>
        AUD {d.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}/yr
      </div>
      <div style={{ color: "var(--muted)" }}>{d.saving.toFixed(2)}% saving</div>
    </div>
  );
}

function HouseholdTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="card" style={{ margin: 0, padding: "10px 14px" }}>
      <strong>HH {d.hh}</strong>{" "}
      <span style={{ color: "var(--muted)" }}>· Cluster {d.cluster}</span>
      <div>
        AUD {d.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}/yr
      </div>
      <div style={{ color: "var(--muted)" }}>
        {d.saving.toFixed(2)}% vs no battery
      </div>
    </div>
  );
}

export default function Overview() {
  const {
    ALGO_MEANS,
    SAVING_PCT,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    ALL_ALGOS,
    SEEDS,
    SEED_HH_DATA,
    HOUSEHOLDS,
    HH_DATA,
  } = useDashboardData();
  const data = ALL_ALGOS.map((a) => ({
    agent: a,
    cost: ALGO_MEANS[a],
    saving: SAVING_PCT[a],
  })).sort((x, y) => x.cost - y.cost);

  // ── Per-household breakdown for a selected model + seed ──
  const [model, setModel] = useState("I-TD3");
  const [seed, setSeed] = useState(1);

  const seedData = SEED_HH_DATA[seed];
  const hhData = HOUSEHOLDS.map((hh) => {
    const costs = seedData[hh];
    const cost = costs[model];
    const base = costs.no_battery;
    return {
      hh,
      cost,
      cluster: HH_DATA[hh]?.cluster ?? "–",
      saving: base > 0 ? ((base - cost) / base) * 100 : 0,
    };
  });
  const meanCost = hhData.reduce((s, d) => s + d.cost, 0) / hhData.length;
  const meanSaving = hhData.reduce((s, d) => s + d.saving, 0) / hhData.length;

  // Best RL agent by mean per-household saving vs no battery (across all households).
  const rlAgents = ALL_ALGOS.filter((a) => a !== "no_battery" && a !== "rule_based");
  const agentSaving = (a) => {
    const vals = HOUSEHOLDS.map((hh) => {
      const c = HH_DATA[hh]?.costs;
      if (!c || c[a] == null || !c.no_battery) return null;
      return ((c.no_battery - c[a]) / c.no_battery) * 100;
    }).filter((v) => v != null);
    return vals.reduce((s, v) => s + v, 0) / vals.length;
  };
  const best = rlAgents
    .map((a) => ({ agent: a, saving: agentSaving(a) }))
    .sort((x, y) => y.saving - x.saving)[0];

  return (
    <div className="page">
      <h2>Overview</h2>
      <p className="subtitle">
        Mean annual electricity cost across all {HOUSEHOLDS.length} evaluated households, by agent.
      </p>

      <div className="hero">
        <div className="label">Best agent</div>
        <div className="big">{best.agent} — {best.saving.toFixed(2)}% saving</div>
        <div className="sub">
          Mean of per-household percentage savings across {HOUSEHOLDS.length} households
        </div>
      </div>

      {/* <div className="card">
        <h3>Mean annual cost by agent (sorted ascending)</h3>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 50 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.06)" />
            <XAxis dataKey="agent" angle={-45} textAnchor="end" interval={0} height={70}
              tick={{ fontSize: 12, fill: '#73726c' }} />
            <YAxis tick={{ fontSize: 12, fill: '#73726c' }}
              label={{ value: 'AUD / year', angle: -90, position: 'insideLeft', fill: '#73726c', fontSize: 12 }} />
            <Tooltip content={<CostTooltip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
            <Bar dataKey="cost" radius={[4, 4, 0, 0]}>
              {data.map(d => <Cell key={d.agent} fill={STRATEGY_COLORS[d.agent]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="legend-strip">
          {GROUPS.map(g => (
            <div className="legend-item" key={g.name}>
              <span className="swatch" style={{ background: g.color }} />
              {g.name}
            </div>
          ))}
        </div>
      </div> */}

      <div className="card">
        <div className="row-between" style={{ marginBottom: 14 }}>
          <h3 style={{ margin: 0 }}>
            Per-household cost — pick a model &amp; seed
          </h3>
          <div style={{ display: "flex", gap: 12 }}>
            <div className="field" style={{ minWidth: 130 }}>
              <label htmlFor="ov-model">Model</label>
              <select
                id="ov-model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {ALL_ALGOS.map((a) => (
                  <option key={a} value={a}>
                    {a} · {STRATEGY_LABELS[a]}
                  </option>
                ))}
              </select>
            </div>
            <div className="field" style={{ minWidth: 90 }}>
              <label htmlFor="ov-seed">Seed</label>
              <select
                id="ov-seed"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              >
                {SEEDS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <p className="subtitle" style={{ margin: "0 0 12px" }}>
          {model} · seed {seed} — mean across {HOUSEHOLDS.length} households: AUD{" "}
          {meanCost.toFixed(0)} · {meanSaving.toFixed(1)}% saving vs no battery
        </p>

        <ResponsiveContainer width="100%" height={360}>
          <BarChart
            data={hhData}
            margin={{ top: 10, right: 20, left: 10, bottom: 40 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="rgba(0,0,0,0.06)"
            />
            <XAxis
              dataKey="hh"
              angle={-45}
              textAnchor="end"
              interval={0}
              height={56}
              tick={{ fontSize: 11, fill: "#73726c" }}
              label={{
                value: "Household",
                position: "insideBottom",
                offset: -2,
                fontSize: 11,
                fill: "#73726c",
              }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: "#73726c" }}
              label={{
                value: "AUD / year",
                angle: -90,
                position: "insideLeft",
                fill: "#73726c",
                fontSize: 12,
              }}
            />
            <Tooltip
              content={<HouseholdTooltip />}
              cursor={{ fill: "rgba(0,0,0,0.04)" }}
            />
            <Bar
              dataKey="cost"
              radius={[4, 4, 0, 0]}
              fill={STRATEGY_COLORS[model]}
            />
          </BarChart>
        </ResponsiveContainer>

        <p className="subtitle" style={{ margin: "22px 0 12px" }}>
          Per-household saving vs no battery (%)
        </p>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart
            data={hhData}
            margin={{ top: 10, right: 20, left: 10, bottom: 40 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="rgba(0,0,0,0.06)"
            />
            <XAxis
              dataKey="hh"
              angle={-45}
              textAnchor="end"
              interval={0}
              height={56}
              tick={{ fontSize: 11, fill: "#73726c" }}
              label={{
                value: "Household",
                position: "insideBottom",
                offset: -2,
                fontSize: 11,
                fill: "#73726c",
              }}
            />
            <YAxis
              unit="%"
              tick={{ fontSize: 12, fill: "#73726c" }}
              label={{
                value: "% saving vs no battery",
                angle: -90,
                position: "insideLeft",
                fill: "#73726c",
                fontSize: 12,
              }}
            />
            <Tooltip
              content={<HouseholdTooltip />}
              cursor={{ fill: "rgba(0,0,0,0.04)" }}
            />
            <Bar
              dataKey="saving"
              radius={[4, 4, 0, 0]}
              fill={STRATEGY_COLORS[model]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

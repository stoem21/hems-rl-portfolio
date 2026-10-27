import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid, ReferenceLine,
} from 'recharts'
import { useDashboardData } from '../data/DashboardDataContext.jsx';

const CLUSTER_COLORS = ['#378ADD', '#1D9E75', '#639922', '#D85A30']

export default function HouseholdDrilldown() {
  const {
    HH_DATA,
    HOUSEHOLDS,
    ALL_ALGOS,
    STRATEGY_COLORS,
  } = useDashboardData();
  const [hh, setHh] = useState(1)
  const entry = HH_DATA[hh]
  const baseline = entry.costs.no_battery

  const data = ALL_ALGOS.map(a => ({
    agent: a,
    cost: entry.costs[a],
    saving: baseline > 0 ? ((baseline - entry.costs[a]) / baseline) * 100 : 0,
  }))

  const Tip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null
    const d = payload[0].payload
    return (
      <div className="card" style={{ margin: 0, padding: '10px 14px' }}>
        <strong>{d.agent}</strong>
        <div>AUD {d.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
        <div style={{ color: 'var(--muted)' }}>{d.saving.toFixed(2)}% vs no battery</div>
      </div>
    )
  }

  return (
    <div className="page">
      <h2>Household drilldown</h2>
      <p className="subtitle">Per-household annual cost for every agent.</p>

      <div className="card">
        <div className="row-between" style={{ marginBottom: 16 }}>
          <div className="field" style={{ maxWidth: 220 }}>
            <label htmlFor="hh">Household</label>
            <select id="hh" value={hh} onChange={e => setHh(Number(e.target.value))}>
              {HOUSEHOLDS.map(h => <option key={h} value={h}>HH {h}</option>)}
            </select>
          </div>
          <span className="pill" style={{ background: CLUSTER_COLORS[entry.cluster] }}>
            Cluster {entry.cluster}
          </span>
        </div>

        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 50 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.06)" />
            <XAxis dataKey="agent" angle={-45} textAnchor="end" interval={0} height={70}
              tick={{ fontSize: 12, fill: '#73726c' }} />
            <YAxis tick={{ fontSize: 12, fill: '#73726c' }}
              label={{ value: 'AUD / year', angle: -90, position: 'insideLeft', fill: '#73726c', fontSize: 12 }} />
            <Tooltip content={<Tip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
            <ReferenceLine y={baseline} stroke="#D85A30" strokeDasharray="6 4"
              label={{ value: 'No battery baseline', position: 'insideTopRight', fill: '#D85A30', fontSize: 11 }} />
            <Bar dataKey="cost" radius={[4, 4, 0, 0]}>
              {data.map(d => <Cell key={d.agent} fill={STRATEGY_COLORS[d.agent]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

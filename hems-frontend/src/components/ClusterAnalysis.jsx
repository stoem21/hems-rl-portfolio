import { useState } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell,
} from 'recharts'
import { useDashboardData } from '../data/DashboardDataContext.jsx';

const CLUSTER_COLORS = ['#378ADD', '#1D9E75', '#639922', '#D85A30']
const RADAR_METRICS = [
  { key: 'rule_based', label: 'Rule-based' },
  { key: 'R-TD3', label: 'No Forecast TD3' },
  { key: 'V-TD3', label: 'Vanilla TD3' },
  { key: 'N-TD3', label: 'Noise TD3' },
  { key: 'I-TD3', label: 'Interval TD3' },
]

export default function ClusterAnalysis() {
  const {
    CLUSTERS,
    TEST_HOUSEHOLDS,
    SEEDS,
    SEED_HH_DATA,
    ALL_ALGOS,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
  } = useDashboardData();
  // Mean cost over the households of one cluster, for a given seed + agent.
  const clusterMean = (seed, cluster, agent) => {
    const ids = TEST_HOUSEHOLDS[String(cluster)]
    const sum = ids.reduce((s, hh) => s + SEED_HH_DATA[seed][hh][agent], 0)
    return sum / ids.length
  }
  const clusterSaving = (seed, cluster, agent) => {
    const base = clusterMean(seed, cluster, 'no_battery')
    return base > 0 ? ((base - clusterMean(seed, cluster, agent)) / base) * 100 : 0
  }


  const [model, setModel] = useState('I-TD3')
  const [seed, setSeed] = useState(1)

  // Radar: saving % per cluster for the 4 reference strategies, at the selected seed.
  const radarData = RADAR_METRICS.map(m => {
    const row = { metric: m.label }
    CLUSTERS.forEach(c => { row[`c${c}`] = Number(clusterSaving(seed, c, m.key).toFixed(2)) })
    return row
  })

  // Per-cluster cost for the selected model + seed.
  const barData = CLUSTERS.map(c => ({
    cluster: `Cluster ${c}`,
    cost: clusterMean(seed, c, model),
    saving: clusterSaving(seed, c, model),
  }))

  return (
    <div className="page">
      <h2>Cluster analysis</h2>
      <p className="subtitle">Households are grouped into 4 behavioural clusters. Pick a model &amp; seed to compare.</p>

      <div className="card" style={{ paddingBottom: 8 }}>
        <div className="row-between">
          <h3 style={{ margin: 0 }}>Selected: {model} · seed {seed}</h3>
          <div style={{ display: 'flex', gap: 12 }}>
            <div className="field" style={{ minWidth: 130 }}>
              <label htmlFor="cl-model">Model</label>
              <select id="cl-model" value={model} onChange={e => setModel(e.target.value)}>
                {ALL_ALGOS.map(a => (
                  <option key={a} value={a}>{a} · {STRATEGY_LABELS[a]}</option>
                ))}
              </select>
            </div>
            <div className="field" style={{ minWidth: 90 }}>
              <label htmlFor="cl-seed">Seed</label>
              <select id="cl-seed" value={seed} onChange={e => setSeed(Number(e.target.value))}>
                {SEEDS.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 20 }}>
        {CLUSTERS.map(c => {
          const base = clusterMean(seed, c, 'no_battery')
          const cost = clusterMean(seed, c, model)
          const sav = clusterSaving(seed, c, model)
          return (
            <div className="stat-card" key={c} style={{ borderTop: `3px solid ${CLUSTER_COLORS[c]}` }}>
              <div className="k">Cluster {c}</div>
              <div className="v">{sav.toFixed(1)}% saving</div>
              <div className="srow"><span>No battery</span><span>AUD {base.toFixed(0)}</span></div>
              <div className="srow"><span>{model}</span><span>AUD {cost.toFixed(0)}</span></div>
            </div>
          )
        })}
      </div>

      <div className="card">
        <h3>{model} cost per cluster · seed {seed}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={barData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.06)" />
            <XAxis dataKey="cluster" tick={{ fontSize: 12, fill: '#73726c' }} />
            <YAxis tick={{ fontSize: 12, fill: '#73726c' }}
              label={{ value: 'AUD / year', angle: -90, position: 'insideLeft', fill: '#73726c', fontSize: 12 }} />
            <Tooltip
              formatter={(v, n, p) => [`AUD ${v.toFixed(0)} · ${p.payload.saving.toFixed(1)}% saving`, model]}
              cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
            <Bar dataKey="cost" radius={[4, 4, 0, 0]} fill={STRATEGY_COLORS[model]}>
              {barData.map((d, i) => <Cell key={i} fill={CLUSTER_COLORS[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Saving % by strategy across clusters · seed {seed}</h3>
        <ResponsiveContainer width="100%" height={420}>
          <RadarChart data={radarData} outerRadius="72%">
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: '#73726c' }} />
            <PolarRadiusAxis angle={90} tick={{ fontSize: 11, fill: '#73726c' }} />
            {CLUSTERS.map(c => (
              <Radar key={c} name={`Cluster ${c}`} dataKey={`c${c}`}
                stroke={CLUSTER_COLORS[c]} fill={CLUSTER_COLORS[c]} fillOpacity={0.12} />
            ))}
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

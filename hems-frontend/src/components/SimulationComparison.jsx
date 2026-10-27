import { useMemo } from "react";
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { simulationKey } from "../lib/simulationStream.js";

const COLORS = ["#1D9E75", "#378ADD", "#D85A30", "#a678de", "#b38b13", "#db5a94", "#3b9eaa", "#6b7632", "#8b603c", "#5267bb", "#a44545", "#555"];

export default function SimulationComparison({ results, selected, onSelect }) {
  const chart = useMemo(() => (results[0]?.cost_timeline || []).map((_, i) => {
    const row = { interval: i + 1, baseline: results[0].baseline_cost_timeline[i] };
    results.forEach((r) => { row[simulationKey(r)] = r.cost_timeline[i]; });
    return row;
  }), [results]);
  if (!results.length) return null;
  const first = results[0];
  return <section className="card calculated-results" aria-label="Calculated simulation results">
    <h3>{results.length > 1 ? "Agent comparison" : "Calculated simulation result"}</h3>
    <p className="subtitle">HH {first.household_id} · {first.start_date} · {first.n_days} day(s) · seed {first.seed}. Numbers come from completed simulations. Select a row to view its charts.</p>
    <div className="comparison-table-wrap"><table className="comparison-table">
      <thead><tr><th>Agent</th><th>Period cost</th><th>Annualised cost</th><th>Saving</th><th>Forecast error</th><th>Chart</th></tr></thead>
      <tbody>{results.map((r) => <tr key={simulationKey(r)} className={selected && simulationKey(selected) === simulationKey(r) ? "selected-result" : ""}>
        <th scope="row">{r.agent}</th><td>AUD {r.period_cost.toFixed(2)}</td><td>AUD {r.annual_cost.toFixed(2)}</td><td>{r.saving_pct.toFixed(2)}%</td>
        <td>{r.error_type === "none" ? "none" : `${r.error_type} ${r.error_pct}%`}</td>
        <td><button className="ghost" type="button" aria-pressed={selected && simulationKey(selected) === simulationKey(r)} onClick={() => onSelect(r)}>{selected && simulationKey(selected) === simulationKey(r) ? "Viewing" : "View graph"}</button></td>
      </tr>)}</tbody>
    </table></div>
    <p className="comparison-note">Annualised costs extrapolate this selected window; they are not a full-year evaluation.</p>
    {results.length > 1 && <><h3>Cumulative electricity cost</h3><ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={chart} margin={{ top: 8, right: 15, left: 8, bottom: 15 }}>
        <CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="interval" label={{ value: "30-min interval", position: "insideBottom", offset: -8 }}/><YAxis label={{ value: "AUD", angle: -90, position: "insideLeft" }}/><Tooltip formatter={(v) => Number(v).toFixed(3)}/><Legend wrapperStyle={{ paddingTop: 12 }}/>
        <Line dataKey="baseline" name="No battery" stroke="#777" strokeDasharray="4 3" dot={false}/>
        {results.map((r, i) => <Line key={simulationKey(r)} dataKey={simulationKey(r)} name={r.agent} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={2}/>)}
      </ComposedChart>
    </ResponsiveContainer></>}
  </section>;
}

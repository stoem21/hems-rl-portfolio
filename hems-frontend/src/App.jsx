import { useState, useEffect } from "react";
import "./App.css";
import { apiFetch } from './config';
import { loadDashboardData } from './data/dashboardData.js';
import { DashboardDataContext } from './data/DashboardDataContext.jsx';
import Overview from "./components/Overview";
import HouseholdDrilldown from "./components/HouseholdDrilldown";
import ClusterAnalysis from "./components/ClusterAnalysis";
import RobustnessHeatmap from "./components/RobustnessHeatmap";
import DigitalTwin from "./components/DigitalTwin";

const TABS = [
  { id: "overview", label: "Overview", Component: Overview },
  { id: "household", label: "Household", Component: HouseholdDrilldown },
  // { id: 'clusters',  label: 'Clusters',      Component: ClusterAnalysis },
  { id: "robust", label: "Robustness", Component: RobustnessHeatmap },
  { id: "twin", label: "Simulation", Component: DigitalTwin },
];

export default function App() {
  const [active, setActive] = useState("overview");
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    setError('');
    const timeout = setTimeout(() => controller.abort(), 20000);
    loadDashboardData(apiFetch, controller.signal).then(payload => {
      if (current) setEvaluation(payload);
    }).catch(cause => {
      if (current) setError(cause.name === 'AbortError' ? 'The evaluation request timed out.' : cause.message);
    }).finally(() => clearTimeout(timeout));
    return () => { current = false; clearTimeout(timeout); controller.abort(); };
  }, [attempt]);
  const Active = TABS.find((t) => t.id === active).Component;

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <h1>
            HEMS Research Dashboard{" "}
            <span>— PV + battery RL agents · Ausgrid</span>
          </h1>
          <nav className="tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab ${active === t.id ? "active" : ""}`}
                onClick={() => setActive(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <p role="note" className="portfolio-notice">
        {evaluation ? `Saved research evaluation loaded from the backend · Export updated ${new Date(evaluation.source.updatedAt).toLocaleString()}` : 'Research results are loaded from the backend through the tunnel.'}
      </p>
      <main>
        {evaluation ? <DashboardDataContext.Provider value={evaluation.data}><Active /></DashboardDataContext.Provider> :
          <div className="page" role={error ? 'alert' : 'status'} aria-live="polite">
            <h2>{error ? 'Research results unavailable' : 'Loading research results…'}</h2>
            {error && <><p>{error} Please check the backend connection and try again.</p>
              <button onClick={() => setAttempt(value => value + 1)}>Try again</button></>}
          </div>}

      </main>
    </div>
  );
}

# Home Energy Management — RL and Simulation Code

A code-focused portfolio for battery-dispatch research: a custom reinforcement-learning environment, training/evaluation scripts, a FastAPI simulation service, an AI tool-calling interface and a React dashboard.

**Dashboard charts load saved research evaluation results from the backend through the HTTPS tunnel.** Thesis text, checkpoints, household datasets and measured evaluation exports stay on the Mac and are excluded from Git. The frontend contains no bundled research results or sample chart fallback.

## Live demo

[Open the HEMS dashboard and simulation demo](https://stoem21.github.io/hems-rl-portfolio/)

The dashboard retrieves saved evaluation results from the live backend; simulation requests run on the research host. Interactive features require the Mac backend and HTTPS tunnel to be online. No dataset, checkpoint or measured result export is bundled in this repository.

![HEMS simulation configuration before a run](docs/images/hems-simulation.jpg)

*Simulation setup screen before running an agent. Measured evaluation results are not embedded in this preview.*

| Area | Technologies |
| --- | --- |
| Dashboard | React 19, Vite, Recharts |
| API | Python, FastAPI, Pydantic |
| RL | PyTorch, Tianshou, Gymnasium; TD3, SAC and DDPG loading/training code |
| AI and retrieval | LangGraph, Ollama/optional provider adapter, local corpus retrieval and cache |
| Experiments | Training/evaluation utilities and MLflow integration |

## Run the dashboard

Requires Node.js 22:

```bash
cd hems-frontend
npm ci
npm run dev
```

Start the combined API proxy with its private evaluation export configured before opening the dashboard. In development, `/dashboard-data` is proxied to `http://127.0.0.1:8787/hems/dashboard-data`; simulation routes use FastAPI on port 8000. Without the export/API, the page shows a loading or connection error state with a retry button. It does not replace results with generated values.

A Docker alternative is available from the repository root:

```bash
docker compose up -d --build
```

Set `VITE_BACKEND_URL` in the build environment to the combined HTTPS API base, then open `http://localhost:8180`. This Docker image builds only the frontend and does not copy research assets or start Python inference.

## Connect the live backend

Set the public build variable:

```text
VITE_BACKEND_URL=https://YOUR-ASSIGNED-DOMAIN.ngrok-free.app/hems
```

The combined Mac proxy serves `GET /hems/dashboard-data` from the private file selected by `hemsDashboardDataFile` in its local configuration. Use the existing generated `hems-frontend/src/data/dashboardData.js` on the research host, or an updated export from `gen_dashboard_data.py`. Only an operator-owned, trusted export should be configured: JavaScript exports are loaded as modules. The response includes export modification time and SHA-256; metric values are passed through unchanged. This identifies the saved export, without independently verifying thesis agreement.

The proxy forwards `/hems/simulate`, `/hems/simulate/compare`, `/hems/ask` and related API calls to FastAPI. Runtime data, checkpoints and thesis retrieval stay on the Mac. See [deployment](docs/DEPLOYMENT.md) and [runtime inputs](docs/RUNTIME-INPUTS.md).

## Architecture

```mermaid
flowchart LR
  Web[React dashboard on Pages] --> Proxy[HTTPS tunnel and API proxy]
  Proxy --> Results[Private saved evaluation export]
  Proxy --> API[FastAPI on a separate host]
  API --> RL[Policy loading and simulation]
  API --> Agent[AI tool calling]
  RL --> Artifacts[Local dataset and checkpoints]
  Agent --> Corpus[Local corpus and Ollama]
```

## Inspect the code

- `hems-backend/hems_env.py`: environment, observations, battery dynamics and reward.
- `train.py` / `train_helper.py`: training pipeline and variants.
- `test.py` / `rescore_eval.py`: evaluation utilities; their inputs remain local.
- `hems-backend/simulation_service.py`: validation, rollout and comparisons.
- `hems-backend/rag_agent.py`: tool calling, retrieval and persistent embedding cache.
- `hems-frontend/src/lib/simulationStream.js`: simulation requests and streaming events.

Forecast variants are represented by `I-*`, `N-*`, `V-*` and `R-*` configurations. Displayed rankings and savings come from the configured saved evaluation export. Simulation runs are computed separately by FastAPI; thesis consistency requires comparison with the thesis tables and experiment settings.

## Engineering highlights

- **Battery-dispatch environment:** A custom Gymnasium environment represents household demand, solar generation, battery state and dispatch rewards. Shared training utilities support TD3, SAC and DDPG configurations.
- **Validated simulation windows:** The FastAPI service checks household and agent selections, calendar dates, continuous half-hour measurements and forecast lookahead before a rollout. Missing intervals and incomplete measurements produce explicit errors.
- **Streaming results:** The React client handles fragmented UTF-8 server-sent events and preserves completed simulation results when an answer stream is interrupted. Comparison results remain tied to the same household and date window.
- **Runtime research assets:** The public source contains application and research pipeline code. Datasets, checkpoints, thesis retrieval material and measured evaluation exports remain on the host; the dashboard retrieves results through the API.
- **Request-conscious delivery:** Connection checks are shared and cached for five minutes, with no periodic health polling. A manual check supports reconnecting when the backend becomes available.

## My contribution

I developed this as a solo research and software project with AI coding assistance. My work covers the battery-dispatch RL environment, training and evaluation pipeline, FastAPI simulation service and React research dashboard.

For this public portfolio, I connected the dashboard to the live backend while keeping datasets, checkpoints, thesis material and measured exports on the research host. AI tools assisted with implementation, debugging, tests and documentation. Software checks are documented in the [validation scope](docs/VALIDATION.md); passing them does not independently reproduce the research results or establish agreement with thesis tables.

## Validate without research data or model weights

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
pytest tests -q
```

```bash
cd hems-frontend
npm ci
node --test src/lib/simulationStream.test.js src/data/dashboardData.test.js src/hooks/useBackendHealth.test.cjs
node --test ../ops/tests/dashboard-data.test.cjs
npm run build
```

See [validation scope](docs/VALIDATION.md). Offline tests exercise application logic and mocked services; they do not reproduce training results or benchmark live model quality.

## Publication scope and attribution

Research artifacts are intentionally excluded. `gen_dashboard_data.py` writes regenerated measured outputs under ignored `private-results/`, keeping measured values out of the frontend source bundle. Configure that private export path on the proxy to serve updated results at runtime. Ausgrid preprocessing code is included; dataset distribution is not.

This repository does not include an open-source license for the application. Third-party packages retain their own licenses. The preprocessing pipeline targets the [Ausgrid Solar Home Electricity Data](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data); users obtain the dataset from its source under the applicable terms.

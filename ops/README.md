# One tunnel for two local backends

Copy `portfolio-ngrok.example.json` to `portfolio-ngrok.json` locally, fill the absolute paths to your existing Clinic demo and HEMS backend runtime checkouts, and set the exact GitHub Pages origin in `allowedOrigins`. Add the exact local frontend origin when using a preview: normally `http://localhost:5173` for Vite or `http://localhost:8180` for the Docker frontend. A different scheme, host or port is a different origin.

The Clinic runtime checkout must provide `compose.yaml` with service `clinic-demo`; the HEMS runtime checkout must provide `docker-compose.yml` with service `api`. This public repository's own Compose file is a frontend preview, not a full inference deployment. The runtime may use private artifacts that are absent from this repository.

Run `bash ops/start-portfolio-ngrok.command --check`, then `bash ops/start-portfolio-ngrok.command` from the repository root. For an already running backend, use `--no-start-backends`. Ctrl+C stops the tunnel/proxy while retaining containers.

The API prefixes are `/clinic/api/...` and `/hems/...`. The proxy binds localhost and supports credentialed CORS, streamed AI responses and a configurable heavy-job limit. It does not host the frontend applications.

ngrok account authentication stays in the local ngrok CLI configuration; it is not a frontend token. Set the proxy port and upstream ports to match the actual host ports. Generated runtime logs/lock files go under ignored `work/portfolio-ngrok`.

Set `hemsDashboardDataFile` to an absolute path to the private generated evaluation export (for example the research runtime's `hems-frontend/src/data/dashboardData.js`). `GET /hems/dashboard-data` serves those actual saved values with a timestamp/hash; unavailable data returns 503. Do not copy the export into this repository. JavaScript exports must be trusted operator-owned files. Restart this launcher after proxy/config edits.

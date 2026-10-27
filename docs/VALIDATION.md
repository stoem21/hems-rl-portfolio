# Validation status

Last reviewed **28 October 2026**. [GitHub Actions](https://github.com/stoem21/hems-rl-portfolio/actions/workflows/ci.yml) reports checks for each published revision.

| Check | Verified result |
| --- | --- |
| Python backend tests | 66 passed in GitHub Actions |
| Frontend and evaluation-proxy Node tests | 20 passed locally; the same suites run in GitHub Actions |
| Frontend production build | Passed locally with the Pages base path and in GitHub Actions |
| Fresh npm dependency installation | Verified locally and in GitHub Actions |
| Fresh inference/training environment installation | Not reproduced; runtime requirements depend on the existing checkpoints and host |
| Docker frontend preview build | Not executed as part of portfolio validation |
| Saved evaluation API | All 26 returned fields matched the configured host export in the recorded live check; timestamp/hash provenance was also checked |
| Short real simulation | I-TD3, seed 1, household 1, one day: validation and rollout passed through the public tunnel |
| Published browser dashboard | Actual saved results loaded; the Simulation page connected to the backend |
| Live AI, full comparison workflows and retraining | Further runtime validation required |

The Node total comprises 7 simulation/stream tests, 3 dashboard-data contract tests, 6 health-hook tests and 4 evaluation-proxy tests. Python and Node tests use fixtures or mocked dependencies where appropriate. They verify software behavior, not live Ollama quality, scientific performance or agreement with thesis tables.

Additional gateway in-memory checks and launcher mock smoke checks were recorded during initial preparation. These are separate from the public repository's test commands above. A new inference host still needs the local inputs described in [runtime inputs](RUNTIME-INPUTS.md).

Live checks were recorded on 27–28 October 2026 against the existing backend. The public dashboard requests saved evaluation results through the proxy and displays export provenance. Original research artifacts remain on the host; they were not modified by these checks. A file modification timestamp identifies the saved export and is not an experiment execution timestamp.

## Health request budget

The hook checks on entry to Simulation, shares pending/recent results across remounts, and has no periodic polling. Returning to a visible tab can refresh a result older than five minutes. Check connection explicitly retries even within that cache period. Hidden tabs do not initiate health checks.

Deterministic hook tests cover 30 idle minutes, rapid tab switching, remounts, offline retries and timeouts. In the live browser check, three Overview/Simulation round trips issued zero additional health requests; an explicit connection check issued one successful health GET.

```bash
cd hems-frontend
node --test src/hooks/useBackendHealth.test.cjs
```

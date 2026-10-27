# Validation of this prepared source copy

Validation recorded during portfolio preparation, 23–27 October 2026. GitHub Actions supplies the validation status for published revisions.

| Check | Result |
| --- | --- |
| Python backend tests | 66 passed |
| Frontend simulation/stream and evaluation-loading tests | Passed (10 tests) |
| Vite production build | Passed using a Pages base path and a placeholder combined-proxy API URL |
| Combined proxy in-memory tests | 12 existing + 4 new evaluation endpoint/loader tests passed |
| Combined launcher mock smoke checks | 6 passed during tunnel preparation |
| Fresh dependency installation from lockfiles | Not executed; available local dependencies were reused |
| Docker frontend preview build | Not executed in this sandbox |
| Live health, saved evaluation export and public tunnel | Passed against the existing backend through a temporary combined tunnel route |
| Short real simulation | I-TD3, seed 1, household 1, one day: validation and rollout passed through the public tunnel |
| Browser dashboard | Overview and Robustness rendered actual evaluation values in the Pages-style local build |
| Live AI streams, comparison and complete interactive flows | Further runtime review required |
| RL retraining and model-quality evaluation | Not performed; actual artifacts/results are excluded from Git |

Offline tests cover API logic, environment rules, validation, stream request handling and RAG/cache logic using test fixtures/mocks. They do not prove live Ollama behavior or scientific performance.

The proxy's socket-based integration suite could not run because the sandbox blocks local listening sockets. Its real handler and stream behavior were checked with an in-memory transport, while launcher lifecycle was checked with mock commands.

The dashboard now requests actual saved evaluation results through the combined proxy and shows export provenance. It contains no embedded result dataset. Existing research artifacts on the Mac backend are unchanged. Check `docs/RUNTIME-INPUTS.md` before attempting inference from this source-only checkout.

Updated 27 October 2026: the actual host evaluation export was loaded and all 26 returned data fields compared exactly with the source module. The frontend response contract accepted it, and all five data-consuming components rendered in a server-render smoke check. The Pages production build passed. The combined stream/proxy and frontend data tests passed (26 total across the existing and new Node suites). Clinic setup was checked to generate `DEMO_AUTO_LOGIN=true` while retaining private file permissions and overwrite protection.

The initial checks used local file reads and in-memory requests. They did not establish agreement with thesis tables. No original research file or backend image was modified.

A subsequent live check on 27 October 2026 verified public ngrok access to both backends, all 26 evaluation fields identified by the original export hash, Clinic auto-login, allowed-origin CORS/preflight, and rejection of an unapproved origin. A real I-TD3 rollout for household 1 / seed 1 / one day completed successfully. A local Pages-style browser build displayed the actual Overview and Robustness data. The original standalone tunnel was restored after the temporary routing check, and temporary proxy/preview services were stopped. This does not verify every simulation mode, live AI, scientific reproducibility, or a fresh backend image build.

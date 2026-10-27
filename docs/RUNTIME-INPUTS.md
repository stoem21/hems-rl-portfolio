# Runtime inputs stay on the backend

The public repository contains algorithms and service code, not the files used by the live research deployment. The Mac backend can continue using its existing local files independently of the GitHub repository.

If running this code in another local backend environment, configure these variables before starting FastAPI:

| Variable | Local input |
| --- | --- |
| `HEMS_DATA_PATH` | Normalized CSV with `customer_id`, `dt`, `load`, `pv`, `price` |
| `HEMS_AGENTS_DIR` | Policy checkpoints named like `I-TD3-1.pt` |
| `HEMS_CORPUS_DIR` | Corpus directory containing the expected section files and `my_bib.bib` |
| `OLLAMA_URL` | The existing local Ollama HTTP endpoint |
| `OLLAMA_EMBED_MODEL` | Locally installed embedding model |

The example environment is documentation; export the values in the backend process environment or place backend-specific local configuration where that process loads it. Frontend `VITE_` configuration is separate and must never contain provider credentials.

`python3 scripts/check-runtime.py` checks file existence without reading artifact contents. A source-only checkout is expected to report missing inputs. Full runtime installation may require CPU/MPS-specific PyTorch and compatibility with the checkpoint-producing environment; the existing requirements lock is supplied as a reference. Training/inference dependency installation was not reproduced during this preparation.

Corpus layout expected by the retrieval code:

```text
sections/introduction.tex
sections/related_work.tex
sections/method.tex
sections/experimental.tex
sections/result.tex
sections/conclusion.tex
backpages/my_appendix.tex
my_bib.bib
```

Use only local or authorized research inputs. No artifacts, data downloads, original thesis text, bibliography or saved evaluation tables are bundled. Training's `--clusters` argument takes a local JSON mapping of cluster IDs to household IDs; the derived original cluster assignment file is also excluded.

The combined Node proxy also needs `hemsDashboardDataFile` in its local JSON configuration. It reads the saved evaluation export directly on the Mac and serves it through `/hems/dashboard-data`; this is separate from FastAPI's model/dataset configuration. Generated export files and their values stay out of Git.

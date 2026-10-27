"""RAG + LangGraph tool-calling agent for the /ask endpoint.

Mirrors agent_loader.py's import discipline: everything from langgraph,
langchain_anthropic, langchain_ollama, agent_loader, and main is imported
INSIDE functions, never at module scope. This keeps `import rag_agent`
working in CI (requirements-test.txt has no langgraph/torch) so tests can
monkeypatch stream_agent without ever installing the heavy deps.

Corpus: the project's own thesis chapters (hems-backend/thesis_corpus/),
embedded once into a persistent disk cache; loaded lazily into RAM on retrieval
and evicted after configurable idle time (15 minutes by default).
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from time import monotonic
import zipfile

import numpy as np

_CORPUS_DIR = os.environ.get("HEMS_CORPUS_DIR", os.path.join(os.path.dirname(__file__), "thesis_corpus"))

_SECTION_FILES = [
    "sections/introduction.tex",
    "sections/related_work.tex",
    "sections/method.tex",
    "sections/experimental.tex",
    "sections/result.tex",
    "sections/conclusion.tex",
    "backpages/my_appendix.tex",
]
_BIB_FILE = "my_bib.bib"

_MAX_CHUNK_CHARS = 900


def _parse_bib(bib_path: str) -> dict[str, str]:
    """Regex-based BibTeX parsing (42 well-formed entries — no need for a real
    parser). Maps citation key -> a short readable form, e.g. 'IRENA, 2024'."""
    text = open(bib_path, encoding="utf-8").read()
    entries = re.findall(r"@\w+\{\s*([^,\s]+)\s*,(.*?)\n\}", text, flags=re.S)
    bib_map: dict[str, str] = {}
    for key, body in entries:
        author_m = re.search(r"author\s*=\s*\{+(.*?)\}+\s*,", body, flags=re.S)
        year_m = re.search(r"year\s*=\s*\{(.*?)\}", body)
        year = year_m.group(1).strip() if year_m else "n.d."
        if author_m:
            first_author = re.split(r"\s+and\s+", author_m.group(1))[0]
            # "Last, First" or "First Last" -> just the surname-ish first token
            surname = first_author.split(",")[0].split()[-1] if first_author.strip() else key
            label = f"{surname} et al., {year}" if " and " in author_m.group(1) else f"{surname}, {year}"
        else:
            label = f"{key}, {year}"
        bib_map[key] = label
    return bib_map


def _resolve_cites(text: str, bib_map: dict[str, str]) -> str:
    def _sub(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        labels = [bib_map.get(k, k) for k in keys]
        return f"({'; '.join(labels)})"

    return re.sub(r"\\cite[a-zA-Z]*\{([^}]*)\}", _sub, text)


def _strip_latex(text: str) -> str:
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\(ref|eqref)\{[^}]*\}", "[reference omitted]", text)
    text = re.sub(r"\\(textbf|emph|textit|texttt)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\\,", " ", text)
    text = re.sub(r"\\(clearpage|noindent|par)\b", "", text)
    return text


def _drop_comments(text: str) -> str:
    """Drop full-line/trailing '%' comments, but not '\\%' (escaped percent)."""
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(line)
    return "\n".join(lines)


def _extract_tables(text: str, source: str, bib_map: dict[str, str]) -> tuple[list[dict], str]:
    """Pull \\begin{table}...\\end{table} blocks out as single chunks (caption +
    label + rows kept together), before generic stripping can mangle them.
    Returns (table_chunks, remaining_text_with_tables_removed)."""
    chunks = []

    def _replace(m: re.Match) -> str:
        block = m.group(0)
        block = _resolve_cites(block, bib_map)
        block = _strip_latex(block)
        block = re.sub(r"\s+", " ", block).strip()
        chunks.append({"text": block, "source": source, "kind": "table"})
        return ""

    remaining = re.sub(r"\\begin\{table\}.*?\\end\{table\}", _replace, text, flags=re.S)
    return chunks, remaining


def _split_prose(text: str, source: str) -> list[dict]:
    chunks = []
    # Split on section/subsection boundaries first, then paragraphs.
    parts = re.split(r"\\(?:sub)*section\*?\{[^}]*\}", text)
    for part in parts:
        for para in re.split(r"\n\s*\n", part):
            para = re.sub(r"\s+", " ", para).strip()
            if len(para) < 40:  # skip stray fragments / near-empty splits
                continue
            if len(para) <= _MAX_CHUNK_CHARS:
                chunks.append({"text": para, "source": source, "kind": "prose"})
                continue
            # Split an overlong paragraph at sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + len(sent) + 1 > _MAX_CHUNK_CHARS:
                    chunks.append({"text": buf.strip(), "source": source, "kind": "prose"})
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                chunks.append({"text": buf.strip(), "source": source, "kind": "prose"})
    return chunks


def _clean_and_chunk_tex(path: str, bib_map: dict[str, str]) -> list[dict]:
    """Returns [{"text", "source", "kind": "prose"|"table"}] for one .tex file."""
    raw = open(path, encoding="utf-8").read()
    raw = _drop_comments(raw)
    source = os.path.basename(path)

    table_chunks, remaining = _extract_tables(raw, source, bib_map)

    remaining = _resolve_cites(remaining, bib_map)
    remaining = _strip_latex(remaining)
    prose_chunks = _split_prose(remaining, source)

    return table_chunks + prose_chunks


def _load_corpus_chunks() -> list[dict]:
    bib_map = _parse_bib(os.path.join(_CORPUS_DIR, _BIB_FILE))
    chunks: list[dict] = []
    for rel_path in _SECTION_FILES:
        chunks.extend(_clean_and_chunk_tex(os.path.join(_CORPUS_DIR, rel_path), bib_map))
    return chunks


def _embed(text: str) -> np.ndarray:
    import httpx

    url = f"{os.environ.get('OLLAMA_URL', 'http://localhost:11434').rstrip('/')}/api/embed"
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    r = httpx.post(url, json={"model": model, "input": text}, timeout=30.0)
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


_INDEX_CACHE_VERSION = 1
_INDEX_LOCK = threading.RLock()
_RAG_INDEX: list[dict] | None = None
_RAG_INDEX_KEY: str | None = None
_RAG_LAST_USED = 0.0
_logger = logging.getLogger("hems.rag_cache")


def _cache_path() -> Path:
    return Path(os.environ.get("RAG_CACHE_DIR", str(Path(__file__).parent / ".rag-cache"))) / "thesis-index.npz"


def _index_fingerprint() -> str:
    """Bind vectors to the corpus, chunking format and embedding provider/model."""
    identity = {
        "version": _INDEX_CACHE_VERSION,
        "chunk_chars": _MAX_CHUNK_CHARS,
        "model": os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "provider": os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        "revision": os.environ.get("RAG_EMBEDDING_REVISION", ""),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for relative in [*_SECTION_FILES, _BIB_FILE]:
        digest.update(relative.encode() + b"\0")
        digest.update((Path(_CORPUS_DIR) / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_disk_index(fingerprint: str) -> list[dict] | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            if metadata["fingerprint"] != fingerprint:
                return None
            chunks = metadata["chunks"]
            vectors = saved["vectors"]
            if (not chunks or vectors.dtype != np.float32 or vectors.ndim != 2
                    or vectors.shape[0] != len(chunks) or vectors.shape[1] < 1
                    or not np.isfinite(vectors).all()):
                raise ValueError("Invalid embedding array")
            if not all(isinstance(c, dict) and all(isinstance(c.get(k), str)
                       for k in ("text", "source", "kind")) for c in chunks):
                raise ValueError("Invalid chunk metadata")
            index = [{**chunk, "vector": vector} for chunk, vector in zip(chunks, vectors)]
        _logger.info("RAG index loaded from disk (%d chunks)", len(index))
        return index
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        _logger.warning("RAG disk cache invalid; rebuilding: %s", exc)
        return None


def _write_disk_index(fingerprint: str, chunks: list[dict]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"fingerprint": fingerprint,
                "chunks": [{k: c[k] for k in ("text", "source", "kind")} for c in chunks]}
    vectors = np.stack([c["vector"] for c in chunks]).astype(np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid embeddings returned by provider")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".thesis-index-", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, metadata=json.dumps(metadata), vectors=vectors)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _logger.info("RAG index saved to disk (%d chunks, %d bytes)", len(chunks), path.stat().st_size)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_index() -> list[dict]:
    chunks = _load_corpus_chunks()
    for c in chunks:
        c["vector"] = _embed(c["text"])
    return chunks


def _get_index() -> list[dict]:
    """Load on first retrieval, rebuild stale disk data, and refresh idle expiry.

    The lock covers loading/building so concurrent first requests share one index.
    Callers hold a local reference while searching; eviction cannot invalidate it.
    """
    global _RAG_INDEX, _RAG_INDEX_KEY, _RAG_LAST_USED
    with _INDEX_LOCK:
        fingerprint = _index_fingerprint()
        if _RAG_INDEX is None or _RAG_INDEX_KEY != fingerprint:
            _RAG_INDEX = None
            index = _read_disk_index(fingerprint)
            if index is None:
                index = _build_index()
                try:
                    _write_disk_index(fingerprint, index)
                except OSError as exc:
                    _logger.warning("Could not persist RAG disk cache: %s", exc)
            _RAG_INDEX = index
            _RAG_INDEX_KEY = fingerprint
        _RAG_LAST_USED = monotonic()
        return _RAG_INDEX


def clear_index_memory() -> bool:
    """Drop only the RAM reference; the disk file remains available."""
    global _RAG_INDEX, _RAG_INDEX_KEY, _RAG_LAST_USED
    with _INDEX_LOCK:
        loaded = _RAG_INDEX is not None
        _RAG_INDEX = None
        _RAG_INDEX_KEY = None
        _RAG_LAST_USED = 0.0
        return loaded


def evict_idle_index() -> bool:
    """Release an idle index without waiting for an in-flight load/build."""
    if not _INDEX_LOCK.acquire(blocking=False):
        return False
    try:
        idle_seconds = max(0.01, float(os.environ.get("RAG_INDEX_IDLE_SECONDS", "900")))
        idle = monotonic() - _RAG_LAST_USED
        if _RAG_INDEX is not None and idle >= idle_seconds:
            clear_index_memory()
            _logger.info("RAG index evicted from RAM after %.1f seconds idle; disk cache retained", idle)
            return True
        return False
    finally:
        _INDEX_LOCK.release()


def warm_index() -> int:
    """Explicit maintenance/precompute hook; application startup stays lazy."""
    return len(_get_index())


def _search(query: str, k: int = 4) -> list[dict]:
    # Embed the question first: a slow provider must not hold a local reference
    # to an otherwise idle corpus index. get_index refreshes expiry afterward.
    qvec = _embed(query)
    idx = _get_index()
    scored = sorted(idx, key=lambda c: -_cosine(qvec, c["vector"]))
    # Return text metadata only: tool results must not retain embedding arrays.
    return [{key: c[key] for key in ("text", "source", "kind")} for c in scored[:k]]


def _make_retrieve_tool():
    from langchain_core.tools import tool

    @tool
    def retrieve_thesis_context(query: str) -> str:
        """Search the HEMS thesis (methodology, experiments, results, related work,
        limitations) for passages relevant to `query`. Call this before making claims
        about why the system/agents/experiments behave a certain way, and cite the
        source file when you use it (e.g. 'per method.tex')."""
        results = _search(query, k=4)
        return "\n\n".join(f"[{c['source']} — {c['kind']}]\n{c['text']}" for c in results)

    return retrieve_thesis_context


def _simulation_summary(result: dict) -> dict:
    return {key: result[key] for key in (
        "household_id", "household_role", "household_cluster", "period", "start_date", "n_days", "agent", "seed", "error_type", "error_pct",
        "period_cost", "baseline_period_cost", "annual_cost", "baseline_annual", "saving_pct", "habit", "warnings")}


def _tool_failure(exc):
    from fastapi import HTTPException
    if isinstance(exc, HTTPException):
        detail = exc.detail
        return detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
    return str(exc)


def _make_simulation_tool():
    import asyncio
    from langchain_core.tools import tool
    from simulation_service import SimulateRequest

    @tool(args_schema=SimulateRequest, response_format="content_and_artifact")
    async def run_simulation(household_id: int, start_date: str, n_days: int = 1,
                             agent: str = "I-TD3", seed: int = 1,
                             error_type: str = "none", error_pct: float = 0.0):
        """Run one real RL battery simulation. Dates must be YYYY-MM-DD, days 1-30,
        seeds 1-5 and forecast errors 0-20 percent. Results update the dashboard.
        Use compare_simulations instead when comparing multiple named agents.
        """
        from simulation_service import run_simulation as run, SimulateRequest
        try:
            req = SimulateRequest(household_id=household_id, start_date=start_date, n_days=n_days,
                                  agent=agent, seed=seed, error_type=error_type, error_pct=error_pct)
            result = await asyncio.to_thread(run, req)
            return _simulation_summary(result), {"results": [result], "comparison": False}
        except Exception as exc:
            message = _tool_failure(exc)
            return {"error": message}, {"error": message, "results": []}

    return run_simulation


def _make_comparison_tool():
    import asyncio
    from langchain_core.tools import tool
    from simulation_service import CompareRequest

    @tool(args_schema=CompareRequest, response_format="content_and_artifact")
    async def compare_simulations(household_id: int, start_date: str, agents: list[str],
                                 n_days: int = 1, seed: int = 1,
                                 error_type: str = "none", error_pct: float = 0.0):
        """Compare 2-12 different named RL agents on exactly the same household,
        date window, seed and forecast-error setting. Return actual costs/savings
        for each agent and update the comparison table and chart on the dashboard.
        """
        from simulation_service import compare_simulations as compare, CompareRequest
        try:
            req = CompareRequest(household_id=household_id, start_date=start_date, agents=agents,
                                 n_days=n_days, seed=seed, error_type=error_type, error_pct=error_pct)
            output = await asyncio.to_thread(compare, req)
            return {"results": [_simulation_summary(r) for r in output["results"]]}, output
        except Exception as exc:
            message = _tool_failure(exc)
            return {"error": message}, {"error": message, "results": []}

    return compare_simulations


_GRAPHS: dict[str, object] = {}   # cache keyed by "ollama:qwen3:14b" / "claude:claude-sonnet-4-6" etc.

AGENT_TOOL_NOTE = (
    "\n\nYou have three tools: retrieve_thesis_context (search the thesis; "
    "cite the source file when you use it, e.g. 'per method.tex') and "
    "run_simulation (run a real rollout for a comparison/what-if question — "
    "don't call it for numbers already given in the current context below). "
    "For a numerical comparison of multiple named agents, call compare_simulations "
    "once with those agents and identical household/date/seed settings. Simulation "
    "tools update the charts and return verified numerical summaries; quote only "
    "those numbers. The returned period and household_role are authoritative: "
    "forecast-error settings do not determine whether data is train, validation or test. "
    "Report off-peak charging and peak discharging as distinct quantities; do not "
    "describe all charged energy as discharged during peak. Reply in the language "
    "of the user question. Reuse on-screen settings when the question omits them; if no "
    "on-screen settings exist, ask for the missing household and date rather than "
    "inventing them. If a tool reports an error, explain it and do not invent results. "
    "You are autonomous: when a tool would help answer the question, call it "
    "immediately in the same turn — never ask the user for permission to search "
    "or simulate, and never say 'let me look that up' without actually calling "
    "the tool right then."
)


def _get_graph(engine: str, model: str):
    key = f"{engine}:{model}"
    if key not in _GRAPHS:
        from langgraph.prebuilt import create_react_agent
        from main import EXPLAIN_SYSTEM

        system = EXPLAIN_SYSTEM + AGENT_TOOL_NOTE
        if engine == "claude":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, max_tokens=1024)
        else:  # "ollama"
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=model,
                base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                # Deliberately leave `reasoning` at its default (thinking ON) for
                # Qwen3.x here, unlike /explain's "think": False. Measured directly:
                # with thinking off, qwen3:14b's tool-calling in this ReAct loop was
                # flaky (0-25% of repeated identical questions actually invoked the
                # tool — it would narrate "let me search" without calling it, or
                # decline). With thinking on, tool-calling was 6/6 reliable across
                # two separate test rounds, at the cost of ~45-60s vs ~5-10s per
                # call. Local is the default engine (see /ask's priority in
                # main.py), so this is a deliberate speed-for-correctness trade
                # on the path most people will hit first; Claude stays available
                # and faster as a one-click override.
            )
        tools = [_make_retrieve_tool(), _make_simulation_tool(), _make_comparison_tool()]
        _GRAPHS[key] = create_react_agent(llm, tools=tools, prompt=system)
    return _GRAPHS[key]


def _extract_text(content) -> str:
    """Normalize a chunk's `.content` to plain visible text.

    Ollama (via langchain_ollama) yields plain strings. Claude (via
    langchain_anthropic) yields a LIST of content blocks per chunk — a mix of
    "text" blocks (the visible answer) and "tool_use"/"input_json_delta"
    blocks (the invisible tool-call being composed). Yielding those raw dicts
    into an HTTP stream breaks it silently, so only "text" blocks are kept."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


async def stream_agent_events(question: str, context_block: str, engine: str, model: str):
    """Public text/progress/results stream; full timelines stay out of LLM context."""
    graph = _get_graph(engine, model)
    summarizing = False
    labels = {"retrieve_thesis_context": ("searching", "Searching thesis"),
              "run_simulation": ("simulating", "Running simulation"),
              "compare_simulations": ("comparing", "Comparing agents")}
    async for event in graph.astream_events(
        {"messages": [{"role": "user", "content": f"{context_block}\n\nQuestion: {question}"}]},
        version="v2",
    ):
        kind = event["event"]
        name = event.get("name", "")
        tool_id = str(event.get("run_id", name))
        if kind == "on_tool_start" and name in labels:
            stage, label = labels[name]
            summarizing = False
            yield {"type": "status", "stage": stage, "label": label,
                   "tool": name, "tool_id": tool_id, "state": "running"}
        elif kind == "on_tool_end" and name in labels:
            output = event["data"].get("output")
            artifact = getattr(output, "artifact", None)
            failed = ((isinstance(artifact, dict) and bool(artifact.get("error")))
                      or getattr(output, "status", "") == "error")
            if isinstance(artifact, dict) and artifact.get("results"):
                yield {"type": "simulation", "results": artifact["results"],
                       "comparison": artifact.get("comparison", False)}
            if failed:
                message = artifact["error"] if isinstance(artifact, dict) and artifact.get("error") else str(getattr(output, "content", "Invalid tool input"))
                yield {"type": "error", "message": message}
            yield {"type": "status", "stage": labels[name][0], "label": labels[name][1],
                   "tool": name, "tool_id": tool_id, "state": "failed" if failed else "complete"}
            if not failed:
                yield {"type": "status", "stage": "summarizing", "label": "Summarizing results"}
                summarizing = True
        elif kind == "on_tool_error":
            yield {"type": "error", "message": _tool_failure(event["data"].get("error", "Tool failed"))}
        elif kind == "on_chat_model_stream":
            text = _extract_text(getattr(event["data"]["chunk"], "content", ""))
            if text:
                if not summarizing:
                    yield {"type": "status", "stage": "summarizing", "label": "Writing answer"}
                    summarizing = True
                yield {"type": "text", "text": text}


async def stream_agent(question: str, context_block: str, engine: str, model: str):
    """Keep plain-text /ask clients compatible with the structured stream."""
    async for event in stream_agent_events(question, context_block, engine, model):
        if event["type"] == "text":
            yield event["text"]
        elif event["type"] == "error":
            yield f"(agent unavailable: {event['message']})"

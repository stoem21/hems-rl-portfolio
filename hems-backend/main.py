"""HEMS hybrid dashboard backend.

Serves the built React app (dist/) and the simulation API on a single port.
API routes are declared BEFORE the StaticFiles mount so they take priority.
"""

import asyncio
import json
import logging
import os
import time
import threading
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from simulation_service import (SimulateRequest, CompareRequest, run_simulation,
                                compare_simulations, validate_simulation, summarise_habit,
                                HORIZON_STEPS, STEPS_PER_DAY, DT_HOURS)

from data_config import (
    DATE_DATASET_START, hh_role, hh_cluster, date_period,
)

# Configure logging once. Prefer `logger.info(...)` over print() — it shows in the
# uvicorn terminal with timestamps and respects --log-level / redirect to a file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hems")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Keep the thesis on disk until retrieval, and release idle RAM indexes."""
    from rag_agent import evict_idle_index

    interval = max(0.01, float(os.environ.get("RAG_INDEX_SWEEP_SECONDS", "30")))

    async def cleanup_idle_index():
        while True:
            await asyncio.sleep(interval)
            evict_idle_index()

    cleaner = asyncio.create_task(cleanup_idle_index())
    logger.info("RAG disk cache enabled; lazy RAM loading, idle expiry %s seconds",
                os.environ.get("RAG_INDEX_IDLE_SECONDS", "900"))
    try:
        yield
    finally:
        cleaner.cancel()
        with suppress(asyncio.CancelledError):
            await cleaner


app = FastAPI(title="HEMS Digital Twin", lifespan=_lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every API request: method, path, status code and duration."""
    t0 = time.perf_counter()
    logger.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000
    logger.info("← %s %s %s (%.1f ms)", request.method, request.url.path,
                response.status_code, dt_ms)
    return response


DATA_PATH = os.environ.get("HEMS_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "full_dataset.csv"))

# Cache the dataset in memory on first load.
_df: pd.DataFrame | None = None
_DATA_LOCK = threading.RLock()
logger.info("backend initialised")


def get_df() -> pd.DataFrame:
    """Load the Ausgrid dataset once. Columns: customer_id, dt, load, pv, price."""
    global _df
    with _DATA_LOCK:
        if _df is None:
            df = pd.read_csv(
                DATA_PATH,
                usecols=["customer_id", "dt", "load", "pv", "price"],
                parse_dates=["dt"],
            )
            df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["customer_id"]).copy()
            df["customer_id"] = df["customer_id"].astype(int)
            _df = df
    return _df


# ── API routes (defined BEFORE the static mount) ──

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/household-range")
def household_range(id: int = Query(..., ge=1, le=300)):
    """Return the available date range for a given household."""
    df = get_df()
    hh = df[df["customer_id"] == id]
    if hh.empty:
        raise HTTPException(404, detail={"message": f"No data for household {id}"})
    return {
        "min_date": hh["dt"].min().strftime("%Y-%m-%d"),
        "max_date": hh["dt"].max().strftime("%Y-%m-%d"),
        "role": hh_role(id),
        "cluster": hh_cluster(id),
    }


@app.post("/simulate/validate")
async def simulation_preflight(req: SimulateRequest):
    return await asyncio.to_thread(validate_simulation, req)


@app.post("/simulate")
async def simulate(req: SimulateRequest):
    return await asyncio.to_thread(run_simulation, req)


@app.post("/simulate/compare")
async def simulate_comparison(req: CompareRequest):
    return await asyncio.to_thread(compare_simulations, req)


class ExplainRequest(BaseModel):
    household_id: int
    household_role: str
    period: str
    agent: str
    seed: int
    annual_cost: float
    saving_pct: float
    error_type: str
    error_pct: float
    n_days: int
    start_date: str
    # Usage-habit summary from /simulate (defaulted so older callers still work).
    peak_load_pct: float = 0.0
    solar_cover_pct: float = 0.0
    charge_offpeak_kwh: float = 0.0
    discharge_peak_kwh: float = 0.0
    total_load_kwh: float = 0.0
    # Which explanation engine to use: "ollama" | "claude" | "" (auto/priority).
    engine: str = ""
    # For engine="ollama": which local model (must be in OLLAMA_MODEL_CHOICES).
    model: str = ""


EXPLAIN_SYSTEM = (
    "You are an energy-systems research assistant explaining the result of a "
    "home battery (HEMS) reinforcement-learning simulation to a researcher.\n\n"
    "Key domain vocabulary:\n"
    "- TOU pricing: time-of-use tariff — peak (15:30–21:00, ~0.50 AUD/kWh) costs most; "
    "the battery should charge during cheap off-peak hours (10:30–14:00, ~0.23 AUD/kWh) "
    "and discharge at peak to cut the bill.\n"
    "- Agents: R-* = reactive (no forecast); V-* = vanilla point forecast; "
    "N-* = noise-robust point forecast; I-* = interval forecast (highest information). "
    "TD3 / SAC / DDPG are the RL algorithms.\n"
    "- Saving % = (no-battery cost − agent cost) / no-battery cost × 100.\n"
    "- Train/val period results may be optimistic because the agent was trained on that data.\n"
    "- Forecast error (noise/bias/drift at X%) degrades point-forecast agents more than "
    "R-* (no forecast) or I-* (interval forecast).\n\n"
    "Rules: In 3-5 sentences for a researcher, (1) characterise this household's usage habit from "
    "the peak-load share and solar-coverage figures, (2) explain how the agent cut the bill by "
    "charging the battery in cheap off-peak hours and discharging during the expensive peak, citing "
    "the kWh shifted, and (3) end with one actionable takeaway (e.g. whether a higher-information "
    "agent or more solar self-use would help). Stay grounded in the given numbers; do not invent "
    "figures. Flag if the household is 'train' set or the period is 'train'/'val'."
)

# Optional local LLM via Ollama (used when ANTHROPIC_API_KEY is not set).
OLLAMA_URL = os.environ.get("OLLAMA_URL", "").rstrip("/")
# Local models offered in the explanation selector (first = default). Note: the
# name is OLLAMA_MODEL_CHOICES, not OLLAMA_MODELS — the latter is reserved by
# Ollama for the model storage directory.
OLLAMA_MODEL_CHOICES = [
    m.strip()
    for m in os.environ.get("OLLAMA_MODEL_CHOICES", "qwen3:14b,qwen2.5:7b").split(",")
    if m.strip()
]
OLLAMA_DEFAULT_MODEL = OLLAMA_MODEL_CHOICES[0] if OLLAMA_MODEL_CHOICES else "qwen3:14b"


def _source_headers(source: str) -> dict[str, str]:
    """Header telling the frontend which engine produced the explanation.
    Exposed for CORS so it stays readable behind a cross-origin tunnel."""
    return {
        "X-Explain-Source": source,
        "Access-Control-Expose-Headers": "X-Explain-Source",
    }


def _ask_source_headers(source: str) -> dict[str, str]:
    """Same idea as _source_headers, for /ask — a separate header name since
    Explanation and Ask can be showing different engines at once."""
    return {
        "X-Ask-Source": source,
        "Access-Control-Expose-Headers": "X-Ask-Source",
    }


def _fallback_stream(req: ExplainRequest):
    msg = (
        f"Household {req.household_id} ({req.household_role} set) simulated with "
        f"{req.agent} seed {req.seed} over {req.n_days} day(s) from {req.start_date} "
        f"({req.period} period): annualised cost AUD {req.annual_cost:.0f}, "
        f"saving {req.saving_pct:.1f}% vs no-battery. "
        f"Forecast error: {req.error_type} at {req.error_pct}%. "
    )
    if req.household_role == "train" or req.period in ("train", "val"):
        msg += "Note: this configuration overlaps training/validation data, so the saving may be optimistic. "
    msg += "(Set ANTHROPIC_API_KEY or OLLAMA_URL to enable a model-generated explanation.)"
    for word in msg.split():
        yield word + " "


def _ollama_stream(user_msg: str, model: str):
    """Stream tokens from a locally running Ollama model."""
    import httpx
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        # Disable chain-of-thought for thinking models (e.g. Qwen3.x) so the panel
        # gets the answer, not <think> traces. Ignored by non-thinking models.
        "think": False,
        "messages": [
            {"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "options": {"num_predict": 400},
    }
    try:
        # Generous read timeout: on CPU the first token can take >60 s for a 7B model
        # (prompt eval, plus model load if it was evicted); inter-token gaps are small.
        timeout = httpx.Timeout(300.0, connect=10.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        yield f"(Ollama: {chunk['error']})"
                        break
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except Exception as exc:  # noqa: BLE001
        yield f"(Ollama unavailable: {exc})"


@app.get("/models")
async def models():
    """List the explanation engines the frontend selector can offer. Default is the
    local Ollama model when available (free), otherwise Claude."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    engines = []
    if OLLAMA_URL:
        for m in OLLAMA_MODEL_CHOICES:
            engines.append({"id": f"ollama:{m}", "label": f"Local — {m}",
                            "engine": "ollama", "model": m})
    if api_key:
        cmodel = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        engines.append({"id": "claude", "label": f"Claude — {cmodel}",
                        "engine": "claude", "model": cmodel})
    if OLLAMA_URL and OLLAMA_MODEL_CHOICES:
        default = f"ollama:{OLLAMA_DEFAULT_MODEL}"
    elif api_key:
        default = "claude"
    else:
        default = "fallback"
    return {"engines": engines, "default": default}


@app.post("/explain")
async def explain(req: ExplainRequest):
    user_msg = (
        f"Simulation result:\n"
        f"- Household: {req.household_id} ({req.household_role} set)\n"
        f"- Agent: {req.agent}, seed {req.seed}\n"
        f"- Window: {req.n_days} day(s) from {req.start_date} ({req.period} period)\n"
        f"- Annualised cost: AUD {req.annual_cost}\n"
        f"- Saving vs no-battery: {req.saving_pct}%\n"
        f"- Forecast error injected: {req.error_type} at {req.error_pct}%\n"
    )
    if req.total_load_kwh > 0:
        user_msg += (
            f"- Usage habit: {req.peak_load_pct:.0f}% of demand falls in the expensive peak window; "
            f"rooftop solar covers ~{req.solar_cover_pct:.0f}% of total consumption.\n"
            f"- Battery action: charged {req.charge_offpeak_kwh:.1f} kWh in cheap off-peak hours and "
            f"discharged {req.discharge_peak_kwh:.1f} kWh during peak (the load-shifting that cuts the bill).\n"
        )
    user_msg += "Explain what this means for the researcher."

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Resolve the engine: an explicit request wins; empty falls back to the
    # availability-based priority (Ollama → Claude → template) — local-first,
    # matching /ask's default and /models' own advertised default, so the two
    # never disagree even if this fires before the frontend's /models fetch
    # resolves.
    engine = req.engine or ("ollama" if OLLAMA_URL else ("claude" if api_key else "fallback"))

    if engine == "claude" and api_key:
        def generate():
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                with client.messages.stream(
                    model=model,
                    max_tokens=800,
                    system=EXPLAIN_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                ) as stream:
                    for text in stream.text_stream:
                        yield text
            except Exception as exc:  # noqa: BLE001
                yield f"(explanation unavailable: {exc})"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers=_source_headers(f"Claude ({model})"),
        )

    if engine == "ollama" and OLLAMA_URL:
        omodel = req.model if req.model in OLLAMA_MODEL_CHOICES else OLLAMA_DEFAULT_MODEL
        return StreamingResponse(
            _ollama_stream(user_msg, omodel),
            media_type="text/plain",
            headers=_source_headers(f"Ollama ({omodel})"),
        )

    return StreamingResponse(
        _fallback_stream(req),
        media_type="text/plain",
        headers=_source_headers("Built-in template"),
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    stream_events: bool = False
    # Same optional simulation-context fields as ExplainRequest, so the agent
    # knows what's currently on screen. All default so a context-free ask works.
    household_id: int | None = None
    household_role: str = ""
    period: str = ""
    agent: str = ""
    seed: int | None = None
    annual_cost: float | None = None
    saving_pct: float | None = None
    error_type: str = "none"
    error_pct: float = 0.0
    n_days: int | None = None
    start_date: str = ""
    peak_load_pct: float = 0.0
    solar_cover_pct: float = 0.0
    charge_offpeak_kwh: float = 0.0
    discharge_peak_kwh: float = 0.0
    total_load_kwh: float = 0.0
    # Same engine-selector shape as ExplainRequest.
    engine: str = ""
    model: str = ""


def _format_ask_context(req: AskRequest) -> str:
    if req.household_id is None:
        return "No simulation is currently on screen."
    return (
        f"Current on-screen simulation:\n"
        f"- Household: {req.household_id} ({req.household_role} set)\n"
        f"- Agent: {req.agent}, seed {req.seed}\n"
        f"- Window: {req.n_days} day(s) from {req.start_date} ({req.period} period)\n"
        f"- Annualised cost: AUD {req.annual_cost}\n"
        f"- Saving vs no-battery: {req.saving_pct}%\n"
        f"- Forecast error injected: {req.error_type} at {req.error_pct}%\n"
        f"- Usage habit: {req.peak_load_pct:.0f}% of demand in the peak window; "
        f"solar covers ~{req.solar_cover_pct:.0f}% of consumption.\n"
        f"- Battery action: charged {req.charge_offpeak_kwh:.1f} kWh off-peak, "
        f"discharged {req.discharge_peak_kwh:.1f} kWh at peak.\n"
    )


@app.post("/ask")
async def ask(req: AskRequest):
    # Read fresh (not the frozen OLLAMA_URL module constant) so this responds to
    # env var changes without a restart, matching ANTHROPIC_API_KEY's pattern below.
    ollama_url = os.environ.get("OLLAMA_URL", "").rstrip("/")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    def encode_event(event):
        return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    if not ollama_url and not api_key:
        message = "Set OLLAMA_URL or ANTHROPIC_API_KEY to enable the agent."
        content = ([encode_event({"type": "error", "message": message}),
                    encode_event({"type": "done"})] if req.stream_events else [message])
        return StreamingResponse(iter(content),
            media_type="text/event-stream" if req.stream_events else "text/plain",
            headers=_ask_source_headers("unavailable"))

    # Local-first by default. Measured during development that qwen3:14b's
    # tool-calling was unreliable with thinking mode off (0-25% actual tool
    # calls); with thinking on (rag_agent.py's ChatOllama config) it was 6/6
    # reliable, at ~45-60s/call instead of ~5-10s. Claude stays available and
    # faster — pass {"engine": "claude"} to use it explicitly.
    engine = req.engine or ("ollama" if ollama_url else "claude")
    model = req.model or (
        (os.environ.get("OLLAMA_MODEL_CHOICES", "qwen3:14b,qwen2.5:7b").split(",")[0].strip())
        if engine == "ollama"
        else os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    )

    from rag_agent import stream_agent, stream_agent_events
    context_block = _format_ask_context(req)

    async def generate():
        if req.stream_events:
            yield encode_event({"type": "status", "stage": "thinking", "label": "Planning your request"})
        try:
            if req.stream_events:
                async for event in stream_agent_events(req.question, context_block, engine, model):
                    yield encode_event(event)
            else:
                async for chunk in stream_agent(req.question, context_block, engine, model):
                    yield chunk
        except Exception as exc:  # noqa: BLE001
            if req.stream_events:
                yield encode_event({"type": "error", "message": str(exc)})
            else:
                yield f"(agent unavailable: {exc})"
        if req.stream_events:
            yield encode_event({"type": "done"})

    headers = _ask_source_headers(f"LangGraph agent - {engine}:{model}")
    if req.stream_events:
        headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(generate(), media_type="text/event-stream" if req.stream_events else "text/plain", headers=headers)


# ── Serve the built React app (AFTER the API routes) ──
_DIST = os.path.join(os.path.dirname(__file__), "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {"message": "Run 'npm run build' in hems-frontend to build the React app"}

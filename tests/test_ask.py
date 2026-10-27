"""Tests for /ask — the LangGraph RAG + tool-calling agent endpoint.

/ask streams plain text (like /explain) rather than returning JSON, so these
tests read `r.text` for the body and the `X-Ask-Source` header for which
engine/model answered — same shape as test_api.py's /explain tests.

Never imports langgraph/langchain_ollama/langchain_anthropic directly; the
agent call is mocked via monkeypatch.setattr on rag_agent.stream_agent (an
async generator), same convention as /explain's env-var monkeypatching in
test_api.py. This keeps CI fast and torch/langgraph-free.
"""
from fastapi.testclient import TestClient

import rag_agent
from main import app

client = TestClient(app)


def _fake_stream(chunks, seen=None, engine_key="engine", model_key="model"):
    """Build a fake async-generator replacement for rag_agent.stream_agent that
    yields `chunks` in order and records the call's args into `seen` (if given)."""

    async def _stream(question, context_block, engine, model):
        if seen is not None:
            seen["question"] = question
            seen["context_block"] = context_block
            seen[engine_key] = engine
            seen[model_key] = model
        for c in chunks:
            yield c

    return _stream


def test_ask_without_any_engine(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "How does I-TD3 compare to R-SAC?"})
    assert r.status_code == 200
    assert r.headers["X-Ask-Source"] == "unavailable"
    assert "OLLAMA_URL" in r.text or "ANTHROPIC_API_KEY" in r.text


def test_ask_with_mocked_agent_defaults_to_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama:11434")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    seen = {}
    monkeypatch.setattr(
        rag_agent, "stream_agent",
        _fake_stream(["Mocked answer ", "citing thesis method.tex."], seen),
    )

    r = client.post("/ask", json={"question": "Why does the interval agent do better?"})
    assert r.status_code == 200
    assert r.text == "Mocked answer citing thesis method.tex."
    assert "LangGraph agent" in r.headers["X-Ask-Source"]
    assert seen["engine"] == "ollama"


def test_ask_defaults_to_ollama_when_both_engines_available(monkeypatch):
    """Local-first: with both OLLAMA_URL and ANTHROPIC_API_KEY set and no explicit
    engine in the request, /ask should pick ollama, not claude."""
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama:11434")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    seen = {}
    monkeypatch.setattr(rag_agent, "stream_agent", _fake_stream(["ok"], seen))

    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 200
    assert seen["engine"] == "ollama"


def test_ask_explicit_claude_engine_overrides_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama:11434")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    seen = {}
    monkeypatch.setattr(rag_agent, "stream_agent", _fake_stream(["Claude answer."], seen))

    r = client.post("/ask", json={
        "question": "Compare I-TD3 and R-SAC for household 5",
        "engine": "claude",
        "model": "claude-sonnet-4-6",
    })
    assert r.status_code == 200
    assert seen["engine"] == "claude"
    assert seen["model"] == "claude-sonnet-4-6"
    assert r.text == "Claude answer."


def test_ask_passes_simulation_context(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama:11434")

    seen = {}
    monkeypatch.setattr(rag_agent, "stream_agent", _fake_stream(["ok"], seen))

    r = client.post("/ask", json={
        "question": "What does this mean?",
        "household_id": 7, "household_role": "test", "period": "test",
        "agent": "I-TD3", "seed": 2, "annual_cost": 1000.0, "saving_pct": 20.0,
        "error_type": "none", "error_pct": 0.0, "n_days": 3, "start_date": "2012-08-01",
    })
    assert r.status_code == 200
    assert "7" in seen["context_block"]
    assert "I-TD3" in seen["context_block"]


def test_ask_agent_exception_mid_stream_appears_in_body(monkeypatch):
    """Headers are already committed once streaming starts, so a mid-stream
    exception can't change X-Ask-Source — it shows up as text in the body,
    same as /explain's existing Claude/Ollama branches handle failures."""
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama:11434")

    async def broken_stream(question, context_block, engine, model):
        yield "partial answer... "
        raise RuntimeError("boom")

    monkeypatch.setattr(rag_agent, "stream_agent", broken_stream)

    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 200
    assert "LangGraph agent" in r.headers["X-Ask-Source"]
    assert "partial answer" in r.text
    assert "boom" in r.text

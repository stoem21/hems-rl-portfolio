"""FastAPI endpoint tests — no dataset or torch needed.

/health and /explain-fallback both work without the CSV or ML stack:
  - /health  : pure status response
  - /explain : without ANTHROPIC_API_KEY it returns a deterministic text stream
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


def test_health_timestamp_is_string():
    r = client.get("/health")
    assert isinstance(r.json()["timestamp"], str)


def test_explain_fallback_without_api_key(monkeypatch):
    """Without ANTHROPIC_API_KEY the endpoint streams a deterministic summary."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/explain", json={
        "household_id": 1,
        "household_role": "test",
        "period": "test",
        "agent": "I-TD3",
        "seed": 1,
        "annual_cost": 1200.0,
        "saving_pct": 15.0,
        "error_type": "none",
        "error_pct": 0.0,
        "n_days": 1,
        "start_date": "2012-08-01",
    })
    assert r.status_code == 200
    assert len(r.text) > 20


def test_explain_fallback_mentions_household(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/explain", json={
        "household_id": 42,
        "household_role": "train",
        "period": "train",
        "agent": "N-SAC",
        "seed": 3,
        "annual_cost": 950.0,
        "saving_pct": 22.5,
        "error_type": "none",
        "error_pct": 0.0,
        "n_days": 7,
        "start_date": "2011-03-01",
    })
    assert r.status_code == 200
    # The fallback message always mentions the household id and saving pct
    assert "42" in r.text
    assert "22.5" in r.text

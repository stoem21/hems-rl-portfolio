"""One validated simulation path for the HTTP form and the agent's tools."""
from datetime import date
import logging
import threading
from typing import Literal

import pandas as pd
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_config import date_period, hh_role, hh_cluster

HORIZON_STEPS = 32
STEPS_PER_DAY = 48
DT_HOURS = 0.5
AgentName = Literal['I-TD3', 'I-SAC', 'I-DDPG', 'N-TD3', 'N-SAC', 'N-DDPG',
                    'V-TD3', 'V-SAC', 'V-DDPG', 'R-TD3', 'R-SAC', 'R-DDPG']
ErrorType = Literal['none', 'noise', 'bias_p', 'bias_n', 'drift_p', 'drift_n']
_SIMULATION_LOCK = threading.RLock()
logger = logging.getLogger('hems.simulation')


class SimulationSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    household_id: int = Field(ge=1, le=300, strict=True)
    start_date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    n_days: int = Field(default=1, ge=1, le=30, strict=True)
    seed: int = Field(default=1, ge=1, le=5, strict=True)
    error_type: ErrorType = 'none'
    error_pct: float = Field(default=0, ge=0, le=20, allow_inf_nan=False)

    @field_validator('start_date')
    @classmethod
    def calendar_date(cls, value):
        date.fromisoformat(value)
        return value


class SimulateRequest(SimulationSettings):
    agent: AgentName = 'I-TD3'


class CompareRequest(SimulationSettings):
    agents: list[AgentName] = Field(min_length=2, max_length=12)

    @field_validator('agents')
    @classmethod
    def unique_agents(cls, agents):
        if len(agents) != len(set(agents)):
            raise ValueError('Choose different agents to compare')
        return agents


def _invalid(code, message, **details):
    raise HTTPException(status_code=422, detail={'code': code, 'message': message, **details})


def validated_window(df: pd.DataFrame, req: SimulateRequest) -> pd.DataFrame:
    """Require the exact calendar window and a continuous 16-hour lookahead.

    Reject gaps, duplicates and shortened windows instead of shifting to the next
    available date or silently compressing missing days in the chart.
    """
    hh = df[df['customer_id'] == req.household_id].sort_values('dt').reset_index(drop=True)
    if hh.empty:
        _invalid('household_missing', f'No data is available for household {req.household_id}.')
    start = pd.Timestamp(req.start_date)
    required = req.n_days * STEPS_PER_DAY + HORIZON_STEPS + 1
    expected = pd.date_range(start, periods=required, freq='30min')
    end = expected[-1]
    if start < hh.dt.min() or end > hh.dt.max():
        _invalid('outside_data_range',
                 f'Choose a complete {req.n_days}-day window with 16 hours of forecast lookahead. '
                 f'Household {req.household_id} has data from {hh.dt.min():%Y-%m-%d} to {hh.dt.max():%Y-%m-%d}.',
                 min_date=hh.dt.min().strftime('%Y-%m-%d'), max_date=hh.dt.max().strftime('%Y-%m-%d'),
                 required_end=end.isoformat())
    window = hh[(hh.dt >= start) & (hh.dt <= end)].reset_index(drop=True)
    missing = expected.difference(pd.DatetimeIndex(window.dt))
    if len(missing):
        _invalid('data_gap', f'This window has {len(missing)} missing half-hour intervals '
                 f'(first missing: {missing[0]:%Y-%m-%d %H:%M}). Choose another date range.',
                 missing_intervals=len(missing), first_missing=missing[0].isoformat())
    if window.dt.duplicated().any() or len(window) != required:
        _invalid('irregular_data', 'This window contains duplicate or irregular timestamps. Choose another date range.')
    if not np.isfinite(window[['load', 'pv', 'price']].to_numpy()).all():
        _invalid('invalid_measurements', 'This window contains incomplete measurements. Choose another date range.')
    # Preserve the existing checkpoint observation normalization: the original
    # runner supplies the household's entire suffix, even for short rollouts.
    return hh[hh.dt >= start].reset_index(drop=True)


def validate_simulation(req: SimulateRequest) -> dict:
    from main import get_df

    validated_window(get_df(), req)
    required_end = pd.Timestamp(req.start_date) + pd.Timedelta(minutes=(req.n_days * STEPS_PER_DAY + HORIZON_STEPS) * 30)
    return {'valid': True, 'n_steps': req.n_days * STEPS_PER_DAY,
            'start_date': req.start_date, 'required_end': required_end.isoformat(),
            'message': f'Complete data for {req.n_days} day(s), including forecast lookahead.'}


def run_simulation(req: SimulateRequest) -> dict:
    # Shared policies can be requested from several LangGraph tool workers.
    with _SIMULATION_LOCK:
        return _run_simulation(req)


def compare_simulations(req: CompareRequest) -> dict:
    settings = req.model_dump(exclude={'agents'})
    requests = [SimulateRequest(**settings, agent=agent) for agent in req.agents]
    validate_simulation(requests[0])
    results = [run_simulation(item) for item in requests]
    return {'results': results, 'comparison': True}


def summarise_habit(load, pv, dispatch, rates):
    """Compact 'usage habit + battery action' summary from the rollout timelines,
    so /explain can describe THIS household concretely. Energy in kWh (each step is
    DT_HOURS long); TOU tier inferred from the per-step price."""
    total_load = sum(load) * DT_HOURS
    if total_load <= 0:
        return {"peak_load_pct": 0.0, "solar_cover_pct": 0.0,
                "charge_offpeak_kwh": 0.0, "discharge_peak_kwh": 0.0, "total_load_kwh": 0.0}
    peak_load = sum(l for l, r in zip(load, rates) if r >= 0.45) * DT_HOURS
    solar = sum(pv) * DT_HOURS
    charge_off = sum(d for d, r in zip(dispatch, rates) if d > 0 and r <= 0.27) * DT_HOURS
    discharge_pk = sum(-d for d, r in zip(dispatch, rates) if d < 0 and r >= 0.45) * DT_HOURS
    return {
        "peak_load_pct":      round(peak_load / total_load * 100, 1),
        "solar_cover_pct":    round(solar / total_load * 100, 1),
        "charge_offpeak_kwh": round(charge_off, 2),
        "discharge_peak_kwh": round(discharge_pk, 2),
        "total_load_kwh":     round(total_load, 2),
    }



def _run_simulation(req: SimulateRequest):
    from main import get_df

    warnings = []
    if req.agent.startswith("R-") and req.error_type != "none":
        warnings.append("Forecast error is ignored for reactive agents, which use no forecasts.")
        req = req.model_copy(update={"error_type": "none", "error_pct": 0.0})
    df = get_df()

    window = validated_window(df, req)
    episode_len = req.n_days * STEPS_PER_DAY

    # ── 2. Build env (correct forecast style for the agent) + load checkpoint ──
    # Heavy imports (torch/tianshou) are deferred to here so /health and
    # /household-range stay importable without the ML stack installed.
    from agent_loader import build_sim_env, load_policy, get_action

    try:
        policy = load_policy(req.agent, req.seed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "message": f"Could not load {req.agent} seed {req.seed}: {exc}"}) from exc

    env = build_sim_env(
        df=window,
        agent=req.agent,
        customer_id=req.household_id,
        error_type=req.error_type,
        error_pct=req.error_pct,
        episode_len_steps=episode_len,
    )
    obs, _ = env.reset()

    # ── 3. Run the rollout ──
    soc_timeline, dispatch_timeline, tou_rates = [], [], []
    load_timeline, pv_timeline = [], []
    grid_costs, baseline_costs = [], []

    for step_idx in range(episode_len):
        action = get_action(policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        soc_timeline.append(round(float(info["soc"]), 4))
        # action_mapped_kw: + = charge, - = discharge
        logger.debug("step %d info=%s", step_idx, info)
        dispatch_timeline.append(round(float(info["action_mapped_kw"]), 3))
        tou_rates.append(round(float(info["price"]), 4))
        load_timeline.append(round(float(info["load_kw"]), 3))
        pv_timeline.append(round(float(info["pv_kw"]), 3))
        grid_costs.append(float(info["cost"]))

        # No-battery baseline: pay for net import at the spot price, no battery.
        net = float(info["load_kw"]) - float(info["pv_kw"])
        baseline_costs.append(max(0.0, net) * float(info["price"]) * DT_HOURS)

        if terminated or truncated:
            break

    # ── 4. Costs ──
    n_steps = len(soc_timeline)
    if n_steps != episode_len:
        raise HTTPException(503, detail={'message': 'The simulation ended before the requested window was complete.'})
    actual_days = max(n_steps / STEPS_PER_DAY, 1e-9)
    annual_factor = 365.0 / actual_days

    period_cost = round(sum(grid_costs), 4)
    baseline_period = round(sum(baseline_costs), 4)
    annual_cost = round(period_cost * annual_factor, 2)
    baseline_annual = round(baseline_period * annual_factor, 2)
    saving_pct = (
        round((baseline_annual - annual_cost) / baseline_annual * 100, 2)
        if baseline_annual > 0 else 0.0
    )

    return {
        "warnings": warnings,
        "timestamps": [t.isoformat() for t in window.dt.iloc[:n_steps]],
        "cost_timeline": np.cumsum(grid_costs).round(6).tolist(),
        "baseline_cost_timeline": np.cumsum(baseline_costs).round(6).tolist(),
        "baseline_period_cost": baseline_period,
        "household_id": req.household_id,
        "household_role": hh_role(req.household_id),
        "household_cluster": hh_cluster(req.household_id),
        "start_date": req.start_date,
        "n_days": req.n_days,
        "period": date_period(req.start_date),
        "agent": req.agent,
        "seed": req.seed,
        "error_type": req.error_type,
        "error_pct": req.error_pct,
        "n_steps": n_steps,
        "period_cost": period_cost,
        "annual_cost": annual_cost,
        "baseline_annual": baseline_annual,
        "saving_pct": saving_pct,
        "soc_timeline": soc_timeline,
        "dispatch_timeline": dispatch_timeline,
        "tou_rates": tou_rates,
        "load_timeline": load_timeline,
        "pv_timeline": pv_timeline,
        "habit": summarise_habit(load_timeline, pv_timeline, dispatch_timeline, tou_rates),
    }

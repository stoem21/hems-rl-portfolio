"""Basic HEMSEnv sanity tests — uses a synthetic DataFrame, no CSV or torch needed."""
import numpy as np
import pandas as pd
import pytest

from hems_env import (
    HEMSEnv, BatterySpec, ForecastConfig, ForecastErrorConfig, TimeFeatureConfig,
)

_BATTERY = BatterySpec(
    capacity_kwh=13.5, p_ch_max_kw=4.6, p_dis_max_kw=4.6,
    eta_ch=1.0, eta_dis=1.0, soc_min=0.10, soc_max=0.90,
)
_TIME_CFG = TimeFeatureConfig(include_doy=False, include_tod=True, include_dow=True)


def _make_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": np.ones(n, dtype=int),
        "dt": pd.date_range("2012-01-01", periods=n, freq="30min"),
        "load": np.ones(n, dtype=np.float32),
        "pv": np.zeros(n, dtype=np.float32),
        "price": np.full(n, 0.3, dtype=np.float32),
    })


def _point_env() -> HEMSEnv:
    return HEMSEnv(
        df=_make_df(),
        episode_len_steps=48,
        action_type="continuous",
        allow_grid_charge=True,
        random_start=False,
        start_index=0,
        init_soc=0.5,
        log_enabled=False,
        battery_spec=_BATTERY,
        forecast_cfg=ForecastConfig(
            enabled=True, horizon_hours=4,
            include_pv=True, include_load=True, include_price=True,
            style="point", interval_pct=0.10,
        ),
        forecast_err_cfg=ForecastErrorConfig(enabled=False),
        time_feature_cfg=_TIME_CFG,
        seed=42,
    )


def _none_env() -> HEMSEnv:
    df = _make_df()
    return HEMSEnv(
        df=df,
        episode_len_steps=len(df) - 1,
        action_type="continuous",
        allow_grid_charge=True,
        random_start=False,
        start_index=0,
        init_soc=0.5,
        log_enabled=False,
        battery_spec=_BATTERY,
        forecast_cfg=ForecastConfig(
            enabled=False, horizon_hours=0,
            include_pv=False, include_load=False, include_price=False,
            style="point", interval_pct=0,
        ),
        forecast_err_cfg=ForecastErrorConfig(enabled=False),
        time_feature_cfg=_TIME_CFG,
        seed=42,
    )


def test_reset_obs_shape_matches_observation_space():
    env = _point_env()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_obs_is_float_array():
    env = _point_env()
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype in (np.float32, np.float64)


def test_none_style_obs_smaller_than_point():
    """R-* agents observe no forecasts so their obs must be strictly smaller."""
    obs_point, _ = _point_env().reset()
    obs_none, _ = _none_env().reset()
    assert obs_none.shape[0] < obs_point.shape[0]


def test_step_returns_correct_types():
    env = _point_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    assert isinstance(obs, np.ndarray)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_info_has_required_keys():
    env = _point_env()
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0]))
    for key in ("soc", "cost", "load_kw", "pv_kw", "price"):
        assert key in info, f"Missing key in step info: {key}"


def test_soc_stays_within_battery_bounds():
    env = _point_env()
    env.reset()
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(np.array([1.0]))  # always charge
        assert _BATTERY.soc_min - 1e-6 <= info["soc"] <= _BATTERY.soc_max + 1e-6
        if terminated or truncated:
            break


def test_episode_ends_after_episode_len_steps():
    env = _point_env()
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        steps += 1
        done = terminated or truncated
    assert steps == env.episode_len_steps

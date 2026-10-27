"""agent_loader.py — Tianshou v1.2.0 checkpoint loading + inference.

Written by reading hems-backend/hems_env.py and model_info.txt. main.py only ever touches the three public helpers
below, so it never needs to import torch / tianshou directly:

    build_sim_env(df, agent, customer_id, error_type, error_pct, episode_len_steps)
    load_policy(agent, seed)            -> cached Tianshou policy
    get_action(policy, obs)            -> float action in [-1, 1]

Key facts (verified):
  * obs_dim depends on FORECAST STYLE, not the algorithm:
        I-*  -> interval style -> obs_dim 168
        V-*, N-* -> point style -> obs_dim 104  (noise is a training-only thing)
  * network: Net([256,256]) actor + critic, same for all 9 agents
  * checkpoint dict key holding the weights is "policy"
  * checkpoint files are named exactly like the UI label: e.g. I-TD3-1.pt
"""

import os
import numpy as np
import pandas as pd
import torch
import tianshou as ts
from tianshou.policy import DDPGPolicy, SACPolicy, TD3Policy
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, ActorProb, Critic

from hems_env import (
    HEMSEnv, BatterySpec, ForecastConfig, ForecastErrorConfig, TimeFeatureConfig,
)

_AGENTS_DIR = os.environ.get("HEMS_AGENTS_DIR", os.path.join(os.path.dirname(__file__), "agents"))

# Style map: I-* → interval, N-*/V-* → point, R-* → none (no forecast in obs)
AGENT_STYLE_MAP = {
    "I-TD3": "interval", "I-SAC": "interval", "I-DDPG": "interval",
    "N-TD3": "point",    "N-SAC": "point",    "N-DDPG": "point",
    "V-TD3": "point",    "V-SAC": "point",    "V-DDPG": "point",
    "R-TD3": "none",     "R-SAC": "none",     "R-DDPG": "none",
}
AGENT_ALG_MAP = {
    "I-TD3": "TD3", "I-SAC": "SAC", "I-DDPG": "DDPG",
    "N-TD3": "TD3", "N-SAC": "SAC", "N-DDPG": "DDPG",
    "V-TD3": "TD3", "V-SAC": "SAC", "V-DDPG": "DDPG",
    "R-TD3": "TD3", "R-SAC": "SAC", "R-DDPG": "DDPG",
}

# UI error label -> (env error_type, bias_sign, drift_sign)  [from test_samp_interval.py]
ERROR_VARIANT_MAP = {
    "noise":   ("noise", +1, +1),
    "bias_p":  ("bias",  +1, +1),
    "bias_n":  ("bias",  -1, +1),
    "drift_p": ("drift", +1, +1),
    "drift_n": ("drift", +1, -1),
}

HORIZON_HOURS = 16
HORIZON_STEPS = 32  # = 16h / 0.5h

_policy_cache: dict = {}


def _battery() -> BatterySpec:
    # Thesis R2+R3 config: efficiency = 1.0 (see model_info.txt).
    return BatterySpec(
        capacity_kwh=13.5, p_ch_max_kw=4.6, p_dis_max_kw=4.6,
        eta_ch=1.0, eta_dis=1.0, soc_min=0.10, soc_max=0.90,
    )


def _forecast_cfg(style: str) -> ForecastConfig:
    if style == "none":
        return ForecastConfig(
            enabled=False, horizon_hours=0,
            include_pv=False, include_load=False, include_price=False,
            style="point", interval_pct=0,
        )
    return ForecastConfig(
        enabled=True, horizon_hours=HORIZON_HOURS,
        include_pv=True, include_load=True, include_price=True,
        style=style, interval_pct=0.10,
    )


def _time_cfg() -> TimeFeatureConfig:
    return TimeFeatureConfig(include_doy=False, include_tod=True, include_dow=True)


def build_sim_env(df: pd.DataFrame, agent: str, customer_id: int,
                  error_type: str = "none", error_pct: float = 0.0,
                  episode_len_steps: int | None = None) -> HEMSEnv:
    """Build a HEMSEnv configured for the given agent's forecast style.

    `error_pct` is a percentage (e.g. 10.0); converted to a fraction internally.
    `df` must contain columns: customer_id, dt, load, pv, price.
    """
    style = AGENT_STYLE_MAP[agent]
    # R-* agents observe no forecasts — error injection is meaningless; always disable.
    if style == "none":
        error_type = "none"
        error_pct = 0.0
    enabled = error_type not in (None, "none", "")
    if enabled:
        etype, bsign, dsign = ERROR_VARIANT_MAP[error_type]
    else:
        etype, bsign, dsign = "noise", +1, +1

    frac = float(error_pct) / 100.0
    ferr = ForecastErrorConfig(
        enabled=enabled, random=False, error_types=(etype,),
        pv_error_pct=frac, load_error_pct=frac,
        bias_sign=bsign, drift_sign=dsign, noise_clip=0.50,
    )

    if episode_len_steps is None:
        episode_len_steps = len(df) - 1 if style == "none" else len(df) - (HORIZON_STEPS + 1)

    return HEMSEnv(
        df=df,
        episode_len_steps=episode_len_steps,
        action_type="continuous",
        allow_grid_charge=True,
        random_start=False,
        start_index=0,
        init_soc=0,
        log_enabled=False,
        battery_spec=_battery(),
        forecast_cfg=_forecast_cfg(style),
        forecast_err_cfg=ferr,
        time_feature_cfg=_time_cfg(),
        customer_id=customer_id,
        reward_mode="cost",
        seed=42,
    )


def _build_dummy_env(style: str) -> HEMSEnv:
    """Minimal synthetic env used only to derive obs/act dims for the policy.

    Observation dimensionality depends solely on the config (style, horizon,
    time features), not on the data values — so synthetic rows are fine.
    """
    n = 200
    df = pd.DataFrame({
        "customer_id": np.ones(n, dtype=int),
        "dt": pd.date_range("2012-01-01", periods=n, freq="30min"),
        "load": np.ones(n, dtype=np.float32),
        "pv": np.zeros(n, dtype=np.float32),
        "price": np.full(n, 0.3, dtype=np.float32),
    })
    return HEMSEnv(
        df=df,
        episode_len_steps=100,
        action_type="continuous",
        allow_grid_charge=True,
        random_start=False,
        start_index=0,
        init_soc=0,
        log_enabled=False,
        battery_spec=_battery(),
        forecast_cfg=_forecast_cfg(style),
        forecast_err_cfg=ForecastErrorConfig(enabled=False),
        time_feature_cfg=_time_cfg(),
        customer_id=1,
        reward_mode="cost",
        seed=42,
    )


def _build_policy(env: HEMSEnv, alg: str, device: str = "cpu"):
    """Reconstruct the policy architecture (matches train_interval.build_policy)."""
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    net_a = Net(obs_dim, hidden_sizes=[256, 256], device=device)
    if alg == "SAC":
        actor = ActorProb(net_a, action_shape=act_dim, max_action=1.0,
                          unbounded=False, device=device).to(device)
    else:
        actor = Actor(net_a, action_shape=act_dim, max_action=1.0, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)

    net_c1 = Net(obs_dim + act_dim, hidden_sizes=[256, 256], device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)

    net_c2 = Net(obs_dim + act_dim, hidden_sizes=[256, 256], device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    if alg == "TD3":
        return TD3Policy(
            actor=actor, actor_optim=actor_optim,
            critic=critic1, critic_optim=critic1_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=0.005, gamma=0.99,
            exploration_noise=GaussianNoise(sigma=0.1),
            policy_noise=0.2, noise_clip=0.5, update_actor_freq=2,
            action_space=env.action_space,
        ).to(device)
    if alg == "SAC":
        target_entropy = float(-act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)
        return SACPolicy(
            actor=actor, actor_optim=actor_optim,
            critic=critic1, critic_optim=critic1_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=0.005, gamma=0.99,
            alpha=(target_entropy, log_alpha, alpha_optim),
            action_space=env.action_space,
        ).to(device)
    if alg == "DDPG":
        return DDPGPolicy(
            actor=actor, actor_optim=actor_optim,
            critic=critic1, critic_optim=critic1_optim,
            tau=0.005, gamma=0.99,
            exploration_noise=GaussianNoise(sigma=0.1),
            action_space=env.action_space,
        ).to(device)
    raise ValueError(f"Unknown alg: {alg}")


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch without the weights_only kwarg
        return torch.load(path, map_location="cpu")


def _load_from_registry(agent: str, alg: str, seed: int) -> dict | None:
    """Download a registered model from MLflow, matching on the seed tag.

    Search order:
      1. Latest version whose tag ``seed`` equals the requested seed.
      2. Overall latest version (fallback when no seed-tagged version exists yet).

    Called only when MLFLOW_TRACKING_URI is set. Returns None on any failure
    so the caller can fall back to local .pt files.

    Model names use the agent prefix (I/N/V/R) to avoid collisions between
    N-* and V-* which share the same forecast style but are distinct agents.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        prefix = agent.split("-")[0]          # "I", "N", "V", or "R"
        model_name = f"hems-{prefix}-{alg}"  # e.g. hems-N-TD3
        client = MlflowClient()

        # Try to find a version tagged with this exact seed
        versions = client.search_model_versions(
            f"name='{model_name}' and tags.seed='{seed}'"
        )
        if versions:
            # Prefer the highest version number among seed matches
            best = max(versions, key=lambda v: int(v.version))
            uri = f"models:/{model_name}/{best.version}"
        else:
            # Fall back to the overall latest version
            uri = f"models:/{model_name}/latest"

        local_path = mlflow.artifacts.download_artifacts(uri)
        ckpt_file = os.path.join(local_path, "best.pt") if os.path.isdir(local_path) else local_path
        if os.path.exists(ckpt_file):
            return _torch_load(ckpt_file)
    except Exception:
        pass
    return None


def load_policy(agent: str, seed: int):
    """Load and cache a policy.

    Tries the MLflow model registry first when MLFLOW_TRACKING_URI is set
    (matching on the seed tag), then falls back to the local agents/ directory.
    """
    key = f"{agent}-{seed}"
    if key in _policy_cache:
        return _policy_cache[key]

    if agent not in AGENT_ALG_MAP:
        raise ValueError(f"Unknown agent '{agent}'. Valid: {sorted(AGENT_ALG_MAP)}")

    style = AGENT_STYLE_MAP[agent]
    alg = AGENT_ALG_MAP[agent]

    dummy_env = _build_dummy_env(style)
    policy = _build_policy(dummy_env, alg, device="cpu")

    loaded = None
    if os.environ.get("MLFLOW_TRACKING_URI"):
        loaded = _load_from_registry(agent, alg, seed)

    if loaded is None:
        path = os.path.join(_AGENTS_DIR, f"{agent}-{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        loaded = _torch_load(path)

    state_dict = loaded["policy"] if isinstance(loaded, dict) and "policy" in loaded else loaded
    policy.load_state_dict(state_dict)
    policy.eval()

    _policy_cache[key] = policy
    return policy


def get_action(policy, obs: np.ndarray) -> float:
    """Run one deterministic forward pass. Returns scalar action in [-1, 1]."""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        batch = ts.data.Batch(obs=obs_tensor, info={})
        result = policy(batch)
        action = np.asarray(result.act).reshape(-1)[0]
    return float(np.clip(action, -1.0, 1.0))

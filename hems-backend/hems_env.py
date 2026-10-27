"""HEMS Environment for Gymnasium and Tianshou
================================================

This module defines a photovoltaic (PV)–battery home energy management
system (HEMS) environment compatible with the Gymnasium API. It
implements features suitable for training reinforcement learning agents
with Tianshou (v1.2.0) or other Gym-based frameworks.  Users can
configure battery specifications, forecast horizons, forecast error
models, and time-based observation features.  The environment
processes a pandas DataFrame of historical PV generation, load
profiles, and prices to simulate an episodic time series environment.

Key capabilities
----------------

* Configurable battery model via ``BatterySpec``.
* Configurable forecast horizon and inclusion of PV and/or load
  forecasts via ``ForecastConfig``.  Forecasts can be included as
  point estimates or min/max intervals.
* Optional forecast error injection via ``ForecastErrorConfig``,
  supporting noise, bias, and drift components.
* Dynamic observation vector that can include sin/cos of
  day-of-year (DOY), time-of-day (TOD), and day-of-week (DOW)
  independently via ``TimeFeatureConfig``.
* Continuous or discrete action spaces mapping actions in [-1, 1] to
  charge/discharge power while respecting state-of-charge (SoC)
  constraints and preventing export.
* Data validation and configurable initial SoC and episode start
  indices.
* Optional per-step logging and a helper to return logs as a
  DataFrame.

The environment does **not** depend on Stable-Baselines3 and uses
only ``gymnasium``, ``numpy``, and ``pandas``.

Examples
--------

At the bottom of this file, a minimal self-test routine
``_quick_rollout_example`` provides a sanity check that the
environment can be instantiated and rolled for a few steps.  When
invoked as a script, the module runs this test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union,Iterable
from collections import deque
import gzip
import pickle

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import uuid

@dataclass
class BatterySpec:
    """Specification of battery parameters.

    Attributes
    ----------
    capacity_kwh: float
        Total battery capacity in kWh.
    p_ch_max_kw: float
        Maximum charging power in kW.
    p_dis_max_kw: float
        Maximum discharging power in kW.
    eta_ch: float
        Charging efficiency (fractional).
    eta_dis: float
        Discharging efficiency (fractional).
    soc_min: float
        Minimum allowable state of charge (fractional, 0–1).
    soc_max: float
        Maximum allowable state of charge (fractional, 0–1).
    """

    capacity_kwh: float = 13.5
    p_ch_max_kw: float = 4.6
    p_dis_max_kw: float = 4.6
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.90


@dataclass
class ForecastConfig:
    """Configuration of forecast horizon and inclusion.

    Attributes
    ----------
    horizon_hours: int
        Number of hours ahead to include forecasts. Valid values are
        4, 8, 12, or 16.
    include_pv: bool
        Whether to include PV forecasts in the observation.
    include_load: bool
        Whether to include load forecasts in the observation.
    style: str
        Forecast style: ``"point"`` for raw forecasts or ``"interval"`` to
        include min/max values for each step.  When using interval style,
        the number of observation elements per forecast step doubles.
    interval_pct: float
        Used only when ``style == "interval"``.  Specifies the width of
        the interval as a percentage of the point forecast.  The
        observation will include ``[value*(1 - interval_pct),
        value*(1 + interval_pct)]`` for each forecasted step.
    """

    enabled: bool = True
    horizon_hours: int = 4
    include_pv: bool = True
    include_load: bool = True
    include_price: bool = True
    style: str = "point"
    interval_pct: float = 0.10


@dataclass
class ForecastErrorConfig:
    """Configuration for optional forecast error injection.

    Attributes
    ----------
    enabled: bool
        If ``True``, forecast errors are applied.
    error_types: Tuple[str, ...]
        A tuple of error type names.  Supported types are ``"noise"`` for
        Gaussian multiplicative noise, ``"bias"`` for constant
        multiplicative bias, and ``"drift"`` for linearly increasing
        multiplicative bias across the forecast horizon.
    pv_error_pct: float
        Base error percentage for PV forecasts.
    load_error_pct: float
        Base error percentage for load forecasts.
    bias_sign: int
        Sign (+1 or –1) applied when the ``"bias"`` error type is
        enabled.
    drift_sign: int
        Sign (+1 or –1) applied when the ``"drift"`` error type is
        enabled.
    noise_clip: float
        Maximum absolute magnitude of multiplicative noise; used to
        avoid extremely large or negative forecasts.  Noise values are
        clipped to the range ``[-noise_clip, noise_clip]`` before
        applying.
    """

    enabled: bool = False
    random: bool = False
    error_types: Tuple[str, ...] = ("noise",)
    pv_error_pct: float = 0.10
    load_error_pct: float = 0.10
    bias_sign: int = +1
    drift_sign: int = +1
    noise_clip: float = 0.50


@dataclass
class TimeFeatureConfig:
    """Configuration for optional time-based observation features.

    Attributes
    ----------
    include_doy: bool
        If ``True``, include sine and cosine of the day-of-year in the
        observation.
    include_tod: bool
        If ``True``, include sine and cosine of the time-of-day (minute
        of day / 1440) in the observation.
    include_dow: bool
        If ``True``, include sine and cosine of the day-of-week (0=Mon…
        6=Sun) in the observation.
    """

    include_doy: bool = True
    include_tod: bool = False
    include_dow: bool = False


class HEMSEnv(gym.Env):
    """Gymnasium-compatible PV–Battery HEMS environment.

    The environment simulates energy flows between a PV array, a
    controllable battery, and household load based on a historical
    dataset.  At each step, an agent chooses a normalized action in
    ``[-1, 1]`` corresponding to a charging or discharging request.
    The environment maps this to real power subject to the battery
    specifications and SoC constraints.  Discharge is limited by the
    remaining load (export is disabled) and optional grid-charging
    constraints.  Rewards are the negative electricity cost from grid
    imports.

    Parameters
    ----------
    df: pandas.DataFrame
        Must contain columns ``"dt"``, ``"price"``, ``"load"`` and
        ``"pv"``.  The ``dt`` column must be convertible to
        ``pd.Timestamp``.  ``load`` and ``pv`` are interpreted as
        average power in kW over each time step.
    battery_spec: BatterySpec
        Battery specification dataclass.  Default corresponds to a
        13.5 kWh battery with symmetrical 4.6 kW charge and discharge
        power and 95 % efficiency.
    forecast_cfg: ForecastConfig
        Configuration of the forecast horizon, which signals to
        include (PV and/or load), and whether to produce point or
        interval forecasts.
    forecast_err_cfg: ForecastErrorConfig
        Optional configuration specifying how to inject errors into
        forecasts.  If ``enabled=False``, forecasts are used as-is.
    time_feature_cfg: TimeFeatureConfig
        Toggles for including sine and cosine of day-of-year (DOY),
        time-of-day (TOD), and day-of-week (DOW) in the observation.
    episode_len_steps: int
        Number of steps per episode.  At each reset, the environment
        samples (or uses the provided ``start_index``) to choose a
        contiguous window of length ``episode_len_steps`` within the
        dataset.  Enough data must remain after the end of the window
        to build forecasts.
    dt_hours: float
        Duration of each time step in hours.  Defaults to 0.5 (30
        minutes).  This affects the conversion between power (kW) and
        energy (kWh), and determines how many data rows constitute a
        forecast horizon of a given number of hours.
    action_type: str
        Either ``"continuous"`` or ``"discrete"``.  Continuous
        actions are taken from a Box space in [-1, 1].  Discrete
        actions are evenly spaced in [-1, 1] with ``discrete_bins``
        number of bins.
    discrete_bins: int
        Number of discrete action bins when ``action_type == "discrete"``.
    allow_grid_charge: bool
        If ``False``, charging is limited to available PV surplus (no
        grid charging).  If ``True``, the battery may draw from the
        grid to charge after PV surplus is used.
    init_soc: float or str
        Initial state of charge for each episode.  If a float is
        provided, SoC is clamped to ``[soc_min, soc_max]``.  If
        ``"random"``, SoC is sampled uniformly from the range
        provided in ``init_soc_range`` or from ``[soc_min, soc_max]``.
    init_soc_range: tuple of float, optional
        Used when ``init_soc == "random"``.  Defines the range from
        which to sample the initial SoC.
    random_start: bool
        If ``True``, the start index for each episode is sampled
        uniformly from all valid starting positions.  If ``False`` and
        ``start_index`` is None, the start index is fixed at 0.
    start_index: int, optional
        If provided, the environment always starts at this index.
        Must be within the valid range defined by the dataset length,
        the episode length, and the forecast horizon.  When provided
        and ``random_start`` is True, the value overrides random
        selection.
    seed: int, optional
        Seed for the internal random number generator used for random
        starts, initial SoC, discrete action selection, and forecast
        errors.  If provided, resetting with the same seed yields
        reproducible episodes.
    log_enabled: bool
        If ``True``, the environment logs information about each step
        in ``self.logs``.  Calling ``get_logs_df()`` returns a
        pandas DataFrame of the logs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        battery_spec: BatterySpec = BatterySpec(),
        forecast_cfg: ForecastConfig = ForecastConfig(),
        forecast_err_cfg: ForecastErrorConfig = ForecastErrorConfig(),
        time_feature_cfg: TimeFeatureConfig = TimeFeatureConfig(),
        episode_len_steps: int = 336,
        dt_hours: float = 0.5,
        action_type: str = "continuous",
        discrete_bins: int = 3,
        allow_grid_charge: bool = True,
        init_soc: Union[float, str] = 0.5,
        init_soc_range: Optional[Tuple[float, float]] = None,
        random_start: bool = True,
        start_index: Optional[int] = None,
        seed: Optional[int] = None,
        log_enabled: bool = True,
        log_dir: Optional[str] = None,
        customer_id: Optional[str] = None,
        reward_mode: str = 'curtail', # cost/curtail/saving
        reward_scale: int = 1,
    ) -> None:
        super().__init__()

        # Validate and store the dataset
        self.df = self._validate_and_prepare_df(df)
        self._rebuild_customer_cache()
        self.batt = battery_spec
        self.fc = forecast_cfg
        self.fe = forecast_err_cfg
        self.tf = time_feature_cfg

        self.episode_len_steps = int(episode_len_steps)
        self.dt_hours = float(dt_hours)

        self.action_type = action_type
        self.discrete_bins = int(discrete_bins)
        self.allow_grid_charge = bool(allow_grid_charge)

        self.init_soc = init_soc
        self.init_soc_range = init_soc_range
        self.random_start = bool(random_start)
        self.start_index = start_index

        self.log_enabled = bool(log_enabled)
        self.logs: List[Dict[str, Any]] = []
        self.log_dir = log_dir
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.customer_id = customer_id

        self.initial_seed = seed
        if seed is None:
            self.initial_seed = 2121
        self.rng = np.random.default_rng(self.initial_seed)
        # for plan env
        self._episode_plan = None          # type: Optional[deque]
        self._on_plan_empty = "error"      # "error" | "cycle" | "random" | "pad_last"
        self._plan_template = None         # เก็บไว้สำหรับ mode cycle
        self._cust_arr = self.df["customer_id"].to_numpy()

        self.episode_id = 0
        self.run_uuid = uuid.uuid4()

        # horizon_steps derived from forecast horizon and dt_hours
        if self.fc.horizon_hours not in (0, 4, 8, 12, 16, 24):
            raise ValueError(
                f"horizon_hours must be one of {{4, 8, 12, 16, 24}}, got {self.fc.horizon_hours}"
            )
        if self.dt_hours <= 0:
            raise ValueError("dt_hours must be positive")
        self.horizon_steps = int(round(self.fc.horizon_hours / self.dt_hours))
        if abs(self.horizon_steps * self.dt_hours - self.fc.horizon_hours) > 1e-6:
            raise ValueError(
                "horizon_hours must be an integer multiple of dt_hours."
            )

        # Validate the dataset can support an episode and forecasts
        self._validate_episode_feasibility()

        # Build Gym spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # Runtime state
        self._t0: int = 0  # starting index of current episode
        self._t: int = 0   # time step within the episode
        self._soc: float = 0.5  # current state of charge [0,1]

        self._price_mean = float(self.df["price"].mean())
        self._price_max = float(self.df["price"].max())
        self._price_std  = float(self.df["price"].std()) + 1e-6

        self._load_mean = float(self.df["load"].mean())
        self._load_std  = float(self.df["load"].std()) + 1e-6
        self._pv_mean   = float(self.df["pv"].mean())
        self._pv_std    = float(self.df["pv"].std()) + 1e-6

        self._price_arr = self.df["price"].to_numpy(dtype=np.float32)
        self._load_arr  = self.df["load"].to_numpy(dtype=np.float32)
        self._pv_arr    = self.df["pv"].to_numpy(dtype=np.float32)
        self._dt_arr    = self.df["dt"].to_numpy()

    # -----------------------------
    # Validation and space construction
    # -----------------------------
    def _validate_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the DataFrame has required columns and clean types.

        Raises
        ------
        ValueError
            If required columns are missing, timestamps are unparseable,
            or load/pv/price contain NaNs after conversion.

        Returns
        -------
        pandas.DataFrame
            A copy of the input with parsed timestamps and numeric
            columns.
        """
        required = {"dt", "price", "load", "pv", "customer_id"}
        missing = required - set(df.columns.tolist())
        if missing:
            raise ValueError(
                f"df missing required columns: {sorted(missing)}. Required: {sorted(required)}"
            )

        out = df.copy()
        # Parse dt column
        out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
        if out["dt"].isna().any():
            bad = int(out["dt"].isna().sum())
            raise ValueError(
                f"df['dt'] has {bad} non-parseable timestamps. Please clean/convert before using."
            )
        # Convert numeric columns
        for c in ["price", "load", "pv","customer_id"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if out[c].isna().any():
                bad = int(out[c].isna().sum())
                raise ValueError(
                    f"df['{c}'] has {bad} NaNs after numeric conversion. Please clean data."
                )
        # Reset index for consistent integer indexing
        out = out.reset_index(drop=True)
        return out

    def _validate_episode_feasibility(self) -> None:
        """Ensure there is enough data to support the requested episode length and forecasts.

        A valid episode requires at least ``episode_len_steps + horizon_steps + 1``
        rows of data (the +1 accounts for using the current step plus
        lookahead).  Additionally, if a fixed ``start_index`` is provided,
        it must be within the valid range.
        """
        n = len(self.df)
        need = self.episode_len_steps + self.horizon_steps + 1
        if n < need:
            raise ValueError(
                f"Not enough data rows for episode_len_steps={self.episode_len_steps} "
                f"+ horizon_steps={self.horizon_steps}. Need >= {need} rows, got {n}."
            )
        if self.start_index is not None:
            max_start = n - need
            if not (0 <= self.start_index <= max_start):
                raise ValueError(
                    f"start_index out of range. Must be in [0, {max_start}] but got {self.start_index}."
                )

    def _build_action_space(self) -> spaces.Space:
        """Construct the action space based on the action type."""
        if self.action_type == "continuous":
            return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        if self.action_type == "discrete":
            if self.discrete_bins < 2:
                raise ValueError("discrete_bins must be >= 2 to cover [-1,1].")
            return spaces.Discrete(self.discrete_bins)
        raise ValueError("action_type must be 'continuous' or 'discrete'.")

    def _obs_dim(self) -> int:
        """Compute the dimensionality of the observation space."""
        # Base features: soc, price, load, pv
        dim = 4
        # Add time features: each selected flag adds 2 (sin and cos)
        if self.tf.include_doy:
            dim += 2
        if self.tf.include_tod:
            dim += 2
        if self.tf.include_dow:
            dim += 2
        if self.fc.enabled and self.fc.include_price:
            dim += self.horizon_steps
        # Forecast blocks: one per included signal (pv and/or load)
        blocks = 0
        if self.fc.enabled and self.fc.include_pv:
            blocks += 1
        if self.fc.enabled and self.fc.include_load:
            blocks += 1
        if blocks > 0:
            if self.fc.style == "point":
                dim += blocks * self.horizon_steps
            elif self.fc.style == "interval":
                dim += blocks * self.horizon_steps * 2
            else:
                raise ValueError("forecast style must be 'point' or 'interval'")
        # print("obs dim",dim)
        return dim

    def _build_observation_space(self) -> spaces.Space:
        """Construct a Box observation space with broad bounds."""
        dim = self._obs_dim()
        # Generic bounds; user may normalize externally
        low = np.full((dim,), -1e9, dtype=np.float32)
        high = np.full((dim,), 1e9, dtype=np.float32)
        # SoC is strictly between 0 and 1
        low[0] = 0.0
        high[0] = 1.0
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _rebuild_customer_cache(self) -> None:
        """Build fast lookup: customer_id array + contiguous segments (start,end) per customer."""
        if "customer_id" not in self.df.columns:
            self._cust_arr = None
            self._cust_segments = None
            return

        cust = self.df["customer_id"].to_numpy().astype(np.int32, copy=False)
        self._cust_arr = cust

        # Find boundaries where customer_id changes (assumes sorted by customer_id)
        change = np.flatnonzero(cust[1:] != cust[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(cust)]
        cust_ids = cust[starts]

        # Detect non-contiguous customers (should not happen if sorted properly)
        # If the same cust_id appears in multiple segments => duplicate in cust_ids
        if len(np.unique(cust_ids)) != len(cust_ids):
            raise ValueError(
                "customer_id is not contiguous in full_df. Please sort by ['customer_id','dt'] "
                "so each customer forms one continuous block."
            )

        self._cust_segments = {int(cid): (int(s), int(e)) for cid, s, e in zip(cust_ids, starts, ends)}

    def set_episode_plan(
        self,
        schedule: List[Tuple[int, int]],
        *,
        on_empty: str = "error",
        strict_slice_check: bool = True,
    ) -> None:
        """
        plan: list of dict; each dict may contain:
        - "start_index": int
        - "df": pd.DataFrame   (ถ้าต้องการสลับ df ต่อ episode)
        - "init_soc": float หรือ "random" (ถ้าต้องการกำหนด init soc ต่อ episode)
        on_empty:
        - "error": ถ้าหมดลิสต์แล้ว reset อีก -> raise
        - "cycle": วนกลับไปต้นลิสต์
        - "random": ใช้ start_index=None แล้วให้ _sample_start_index() สุ่ม/0 ตาม config
        - "pad_last": ถ้าหมดลิสต์ให้ใช้ plan ตัวสุดท้ายซ้ำ
        """
        if on_empty not in ("error", "cycle", "random", "pad_last"):
            raise ValueError("on_empty must be one of: error|cycle|random|pad_last")

        if self._cust_arr is None or self._cust_segments is None:
            raise RuntimeError("This env has no customer_id cache. Ensure df has 'customer_id' column.")

        # validate start_index range ถ้ามี
        n = len(self.df)
        need = self.episode_len_steps + self.horizon_steps + 1

        cleaned = []
        for i, (cid, s) in enumerate(schedule):
            cid = int(cid)
            s = int(s)
            if cid not in self._cust_segments:
                raise ValueError(f"schedule[{i}] customer_id={cid} not found in df.")
            seg_s, seg_e = self._cust_segments[cid]  # [seg_s, seg_e)
            # start must be within this customer's segment
            if not (seg_s <= s < seg_e):
                raise ValueError(
                    f"schedule[{i}] start_index={s} not in customer {cid} segment [{seg_s}, {seg_e})."
                )

            end_excl = s + need
            # end must not exceed customer's segment end
            if end_excl > seg_e:
                raise ValueError(
                    f"schedule[{i}] window exceeds customer boundary: "
                    f"cid={cid}, start={s}, need={need}, end_excl={end_excl}, seg_end={seg_e}."
                )

            # also guard global range (mostly redundant if seg_e <= n)
            if end_excl > n:
                raise ValueError(f"schedule[{i}] window exceeds df length: end_excl={end_excl} > n={n}")

            if strict_slice_check:
                # Strongest guarantee: no mixed customer_id inside the window
                if not np.all(self._cust_arr[s:end_excl] == cid):
                    raise ValueError(
                        f"schedule[{i}] contains mixed customer_id inside window. "
                        f"cid={cid}, start={s}, end_excl={end_excl}"
                    )
            cleaned.append((cid, s))

        self._episode_plan = deque(cleaned)
        self._on_plan_empty = on_empty
        self._plan_template = list(cleaned)  # สำหรับ cycle


    def clear_episode_plan(self) -> None:
        self._episode_plan = None
        self._plan_template = None
        self._on_plan_empty = "error"

    # -----------------------------
    # Gym API methods
    # -----------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and sample a new episode start.

        Parameters
        ----------
        seed: int, optional
            If provided, re-seed the internal RNG for this reset.
        options: dict, optional
            Ignored; present for API compatibility.

        Returns
        -------
        obs: numpy.ndarray
            Initial observation of the new episode.
        info: dict
            Auxiliary information, including ``"t0"`` (start index).
        """
        super().reset(seed=seed)
        self.rng = np.random.default_rng(self.initial_seed+self.episode_id)
        if seed is not None:
            self.rng = np.random.default_rng(seed+self.episode_id)
        if self.log_enabled:
            if len(self.logs) > 0 and self.log_dir:
                import os
                os.makedirs(self.log_dir, exist_ok=True)
                with open(f"{self.log_dir}/cust{self.customer_id or str(0)}_{self.run_uuid}_{str(self.episode_id)}.pkl",'wb') as f:
                    pickle.dump(self.logs,f)
            self.logs = []

        if self.fe.enabled and self.fe.random:
            if self.rng.random() < 0.15:
                self.fe.error_types = ()
            else:
                n_type = self.rng.choice([1,2],p=[0.6,0.4])
                chosen = self.rng.choice(['noise','bias','drift'],n_type,replace=False)
                self.fe.error_types = tuple(chosen)
                self.fe.pv_error_pct = self.rng.uniform(0.02,0.20)
                self.fe.load_error_pct = self.rng.uniform(0.02,0.20)
                if 'bias' in chosen:
                    self.fe.bias_sign  = self.rng.choice([-1, +1])
                if 'drift' in chosen:
                    self.fe.drift_sign = self.rng.choice([-1, +1])

        # ---------- NEW: apply episode plan ----------
        plan_item = None
        if self._episode_plan is not None:
            if len(self._episode_plan) == 0:
                if self._on_plan_empty == "error":
                    raise RuntimeError("Episode plan is empty before epoch ends.")
                elif self._on_plan_empty == "cycle":
                    self._episode_plan = deque(self._plan_template)
                elif self._on_plan_empty == "pad_last":
                    # ใช้ตัวสุดท้ายซ้ำ
                    if not self._plan_template:
                        raise RuntimeError("Episode plan template is empty.")
                    self._episode_plan = deque([self._plan_template[-1]])
                elif self._on_plan_empty == "random":
                    self.start_index = None  # fallback ไป _sample_start_index()

            if len(self._episode_plan) > 0:
                plan_item = self._episode_plan.popleft()

        if plan_item is not None:
            # 1) สลับ df ต่อ episode (optional)
            # import pdb;pdb.set_trace()
            # if "df" in plan_item and plan_item["df"] is not None:
            #     self.df = self._validate_and_prepare_df(plan_item["df"])
            #     self._validate_episode_feasibility()

            # # 2) ตั้ง start_index ต่อ episode (optional)
            # if "start_index" in plan_item:
            #     self.start_index = None if plan_item["start_index"] is None else int(plan_item["start_index"])

            # # 3) ตั้ง init_soc ต่อ episode (optional)
            # if "init_soc" in plan_item:
            #     self.init_soc = plan_item["init_soc"]
            if isinstance(plan_item, (tuple, list)):
                # plan_item = (customer_id, start_index)
                _cid, _start = plan_item
                self.start_index = int(_start)
            elif isinstance(plan_item, dict):
                if "start_index" in plan_item:
                    self.start_index = (
                        None if plan_item["start_index"] is None
                        else int(plan_item["start_index"])
                    )
                if "init_soc" in plan_item:
                    self.init_soc = plan_item["init_soc"]
                if "df" in plan_item and plan_item["df"] is not None:
                    self.df = self._validate_and_prepare_df(plan_item["df"])
                    self._validate_episode_feasibility()


        # --------------------------------------------

        self.episode_id += 1
        self._t0 = self._sample_start_index()
        self._t = 0
        self._soc = self._sample_init_soc()
        obs = self._get_obs()
        info: Dict[str, Any] = {"t0": self._t0}
        # print('start index',)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment.

        Parameters
        ----------
        action: Any
            Either a scalar in [-1, 1] if the action space is continuous, or
            an integer index if the action space is discrete.

        Returns
        -------
        obs: numpy.ndarray
            Next observation vector.
        reward: float
            Reward for this step, equal to the negative cost of grid
            import (in currency units).
        terminated: bool
            Always False; episodes terminate only via truncation.
        truncated: bool
            True if the episode length has been reached.
        info: dict
            Auxiliary information about the step and energy flows.
        """
        idx = self._t0 + self._t
        before_soc = self._soc,
        # row = self.df.iloc[idx]

        # Convert action to a scalar in [-1,1]
        # print('action',action)
        a = self._action_to_scalar(action)
        a = float(np.clip(a, -1.0, 1.0))

        # Map normalized action to requested charge/discharge power
        req_ch_kw: float = 0.0
        req_dis_kw: float = 0.0
        if a >= 0.0:
            req_ch_kw = a * self.batt.p_ch_max_kw
        else:
            req_dis_kw = (-a) * self.batt.p_dis_max_kw

        # Read underlying data for the current step
        # pv_kw = float(row["pv"])
        # load_kw = float(row["load"])
        # price = float(row["price"])
        # dt = row["dt"]
        pv_kw   = float(self._pv_arr[idx])
        load_kw = float(self._load_arr[idx])
        price   = float(self._price_arr[idx])
        dt      = self._dt_arr[idx]

        # Compute PV to load directly used
        pv_to_load_kw = min(pv_kw, load_kw)
        remaining_load_kw = load_kw - pv_to_load_kw
        pv_surplus_kw = pv_kw - pv_to_load_kw  # may be >0 if PV surplus

        # Determine how much we can discharge (no export, limited by SoC)
        dis_kw = min(req_dis_kw, remaining_load_kw)
        dis_kw = min(dis_kw, self._max_discharge_power_by_soc())

        batt_to_load_kw = dis_kw
        remaining_load_kw_after_batt = remaining_load_kw - batt_to_load_kw

        # Determine how much we can charge (limited by SoC)
        ch_kw = req_ch_kw
        ch_kw = min(ch_kw, self._max_charge_power_by_soc())
        if not self.allow_grid_charge:
            # Restrict charging to PV surplus only
            ch_kw = min(ch_kw, max(pv_surplus_kw, 0.0))
        # Use PV surplus first, then grid to satisfy the charging request
        pv_to_batt_kw = min(max(pv_surplus_kw, 0.0), ch_kw)
        grid_to_batt_kw = max(ch_kw - pv_to_batt_kw, 0.0)

        # Compute grid imports (no exports allowed)
        grid_import_kw = remaining_load_kw_after_batt + grid_to_batt_kw
        grid_import_kwh = grid_import_kw * self.dt_hours
        cost = grid_import_kwh * price

        # -----------------------------
        # Per-step costs (currency units)
        # -----------------------------
        # Baseline (do-nothing): use PV for load, do not use the battery at all.

        if self.reward_mode == 'cost':
            reward = -cost
        elif self.reward_mode == 'curtail':
            iam_curtail = self.batt.eta_ch * self.batt.eta_dis * 49.57
            curtail_kwh = max(0.0, (pv_surplus_kw - pv_to_batt_kw)*self.dt_hours)
            reward = -cost - (iam_curtail * curtail_kwh)
        elif self.reward_mode == 'saving':
            grid_import_kw_base = max(remaining_load_kw, 0.0)  # remaining load after PV
            grid_import_kwh_base = grid_import_kw_base * self.dt_hours
            cost_agent_t = grid_import_kwh * price
            cost_base_t = grid_import_kwh_base * price
            reward = cost_base_t - cost_agent_t
        elif self.reward_mode == 'composite':
            # R1: core saving signal — measures agent's true contribution
            grid_import_kw_base = max(remaining_load_kw, 0.0)
            cost_base_t = grid_import_kw_base * self.dt_hours * price
            cost_agent_t = grid_import_kwh * price
            r1 = cost_base_t - cost_agent_t

            # price_ratio = price / (self._price_mean + 1e-6)
            # r2 = r1 * price_ratio * 0.5
            # r3 = pv_to_batt_kw * self.dt_hours * price * 0.3

            curtail_kwh = max(0.0, (pv_surplus_kw - pv_to_batt_kw) * self.dt_hours)
            r3 = -curtail_kwh * price * 0.5

            throughput_kwh = (ch_kw + dis_kw) * self.dt_hours
            r4 = -throughput_kwh * 0.01

            reward = r1 + r3 + r4
            # print(f"R1={r1:.4f} R3={r3:.4f} R4={r4:.4f} total={reward:.4f}")
        else:
            raise ValueError("Unknown reward mode")
        # Update SoC
        self._update_soc(charge_kw=ch_kw, discharge_kw=dis_kw)

        # Advance time step
        self._t += 1
        truncated = self._t >= self.episode_len_steps
        terminated = False
        if truncated:
            soc_margin = self._soc - self.batt.soc_min
            stored_kwh = soc_margin * self.batt.capacity_kwh * self.batt.eta_dis
            next_price = float(self._price_arr[idx + 1]) if idx + 1 < len(self.df) else price
            price_premium = max(next_price - self._price_mean, 0.0)
            r5 = stored_kwh * price_premium * 0.5
            reward += r5

        reward = reward*self.reward_scale

        # Compute net battery power for logging: positive = charging, negative = discharging
        action_mapped_kw = ch_kw - dis_kw

        # Build info dict
        info = {
            "dt": dt,
            "idx": idx,
            "before_soc": before_soc,
            "soc": self._soc,
            "action_raw": a,
            "action_mapped_kw": action_mapped_kw,
            "req_ch_kw": req_ch_kw,
            "req_dis_kw": req_dis_kw,
            "charge_kw": ch_kw,
            "discharge_kw": dis_kw,
            "pv_kw": pv_kw,
            "load_kw": load_kw,
            "price": price,
            "grid_import_kw": grid_import_kw,
            "grid_import_kwh": grid_import_kwh,
            "cost": cost,
            "reward": reward,
            "pv_to_load_kw": pv_to_load_kw,
            "pv_to_batt_kw": pv_to_batt_kw,
            "batt_to_load_kw": batt_to_load_kw,
            "grid_to_batt_kw": grid_to_batt_kw,
            "error_type": "+".join(self.fe.error_types) if self.fe.error_types else "clean"
        }

        if self.log_enabled:
            self.logs.append(dict(info))

        # Construct next observation
        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    # -----------------------------
    # Observation construction
    # -----------------------------
    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector for the current time step."""
        idx = self._t0 + self._t
        # row = self.df.iloc[idx]
        # dt = row["dt"]

        soc = self._soc
        # price = float(row["price"])
        # load_kw = float(row["load"])
        # pv_kw = float(row["pv"])
        pv_kw   = float(self._pv_arr[idx])
        load_kw = float(self._load_arr[idx])
        price   = float(self._price_arr[idx])
        dt      = self._dt_arr[idx]

        price_norm = (price - self._price_mean) / self._price_std
        load_norm  = (load_kw - self._load_mean) / self._load_std
        pv_norm    = (pv_kw   - self._pv_mean)   / self._pv_std

        # Base features
        parts: List[np.ndarray] = [np.array([soc, price_norm, load_norm, pv_norm], dtype=np.float32)]
        # Time features
        tf_vec = self._time_features(dt)
        if tf_vec.size > 0:
            parts.append(tf_vec)
        # Forecasts for next horizon_steps (idx+1 .. idx+H)
        if self.fc.enabled and (self.fc.include_pv or self.fc.include_load or self.fc.include_price):
            f_idx0 = idx + 1
            f_idx1 = idx + 1 + self.horizon_steps
            # future = self.df.iloc[f_idx0:f_idx1]
            if self.fc.include_pv:
                # pv_f = future["pv"].to_numpy(dtype=np.float32)
                pv_f = self._pv_arr[f_idx0:f_idx1].copy()
                # import pdb;pdb.set_trace()
                pv_f = self._apply_forecast_error(pv_f, var="pv")
                pv_f = self._format_forecast_block(pv_f, style=self.fc.style)
                pv_f_norm = (pv_f - self._pv_mean)   / self._pv_std
                parts.append(pv_f_norm)
            if self.fc.include_load:
                # load_f = future["load"].to_numpy(dtype=np.float32)
                load_f = self._load_arr[f_idx0:f_idx1].copy()
                load_f = self._apply_forecast_error(load_f, var="load")
                load_f = self._format_forecast_block(load_f, style=self.fc.style)
                load_f_norm = (load_f - self._load_mean) / self._load_std
                parts.append(load_f_norm)
            if self.fc.include_price:
                # price_f = future["price"].to_numpy(dtype=np.float32)
                price_f = self._price_arr[f_idx0:f_idx1].copy()
                price_f_norm = (price_f - self._price_mean) / self._price_std
                parts.append(price_f_norm)
        obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        return obs

    def _format_forecast_block(self, x: np.ndarray, style: str) -> np.ndarray:
        """Format a forecast block according to the configured style."""
        if style == "point":
            return x.astype(np.float32, copy=False)
        if style == "interval":
            if self.fe.enabled and self.fe.random:
                half_w = self.rng.uniform(0.05,0.20)
                x_min = x * (1.0 - half_w)
                x_max = x * (1.0 + half_w)
                # 20% chance: shift both bounds (miscalibrated interval)
                if self.rng.random() < 0.20:
                    shift = self.rng.uniform(0.05, 0.15) * self.rng.choice([-1, +1])
                    x_min = x_min * (1.0 + shift)
                    x_max = x_max * (1.0 + shift)
            else:
                pct = float(self.fc.interval_pct)
                x_min = x * (1.0 - pct)
                x_max = x * (1.0 + pct)
            x_min = np.maximum(x_min, 0.0)
            # Interleave min and max values: [min0, max0, min1, max1, ...]
            out = np.empty((x.shape[0] * 2,), dtype=np.float32)
            out[0::2] = x_min
            out[1::2] = x_max
            return out
        raise ValueError("Unknown forecast style")

    def _apply_forecast_error(self, x: np.ndarray, var: str) -> np.ndarray:
        """Apply configured forecast errors to an array of forecasts.

        Parameters
        ----------
        x: numpy.ndarray
            Array of forecast values (PV or load) to which errors will
            be applied.
        var: str
            Either ``"pv"`` or ``"load"`` indicating which error
            percentages to use.

        Returns
        -------
        numpy.ndarray
            Forecasts after applying multiplicative errors, with
            negative values clipped to zero.
        """
        if not self.fe.enabled:
            return x
        pct = float(self.fe.pv_error_pct if var == "pv" else self.fe.load_error_pct)
        if pct <= 0:
            return x
        y = x.astype(np.float32, copy=True)
        # print(y)
        H = y.shape[0]

        for et in self.fe.error_types:
            name = et.lower().strip()
            if name == "noise":
                noise = self.rng.normal(loc=0.0, scale=pct, size=H).astype(np.float32)
                # noise = np.clip(noise, -self.fe.noise_clip, self.fe.noise_clip)
                y = y * (1.0 + noise)
            elif name == "bias":
                y = y * (1.0 + self.fe.bias_sign * pct)
            elif name == "drift":
                if H <= 1:
                    continue
                k = np.arange(H, dtype=np.float32)
                factor = 1.0 + self.fe.drift_sign * pct * (k / (H - 1))
                y = y * factor
            else:
                raise ValueError(
                    f"Unknown error type: {et}. Use any of ('noise','bias','drift')."
                )
        # Clip negative values
        y = np.maximum(y, 0.0)
        # print(self.fe.error_types,pct)
        return y

    def _time_features(self, dt: pd.Timestamp) -> np.ndarray:
        """Compute time-based features for a timestamp.

        Returns a numpy array of sin/cos pairs for each enabled
        feature (DOY, TOD, DOW).  The order is deterministic: DOY
        first, then TOD, then DOW.
        """
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        feats: List[float] = []
        if self.tf.include_doy:
            doy = int(dt.dayofyear)
            year_days = 366 if dt.is_leap_year else 365
            theta = 2.0 * np.pi * (doy / year_days)
            feats.append(np.sin(theta))
            feats.append(np.cos(theta))
        if self.tf.include_tod:
            minute_of_day = dt.hour * 60 + dt.minute
            theta = 2.0 * np.pi * (minute_of_day / 1440.0)
            feats.append(np.sin(theta))
            feats.append(np.cos(theta))
        if self.tf.include_dow:
            dow = int(dt.dayofweek)  # Monday=0 .. Sunday=6
            theta = 2.0 * np.pi * (dow / 7.0)
            feats.append(np.sin(theta))
            feats.append(np.cos(theta))
        return np.array(feats, dtype=np.float32)

    # -----------------------------
    # Helpers for battery dynamics
    # -----------------------------
    def _max_charge_power_by_soc(self) -> float:
        """Maximum charge power allowed by the current SoC.

        Returns the lesser of the physical maximum (``p_ch_max_kw``) and
        the limit imposed by ``soc_max``.
        """
        headroom = max(self.batt.soc_max - self._soc, 0.0)
        denom = self.dt_hours * max(self.batt.eta_ch, 1e-9)
        kw = headroom * self.batt.capacity_kwh / denom
        return float(min(kw, self.batt.p_ch_max_kw))

    def _max_discharge_power_by_soc(self) -> float:
        """Maximum discharge power allowed by the current SoC.

        Returns the lesser of the physical maximum (``p_dis_max_kw``) and
        the limit imposed by ``soc_min``.
        """
        margin = max(self._soc - self.batt.soc_min, 0.0)
        denom = self.dt_hours
        kw = margin * self.batt.capacity_kwh * max(self.batt.eta_dis, 1e-9) / max(denom, 1e-9)
        return float(min(kw, self.batt.p_dis_max_kw))

    def _update_soc(self, charge_kw: float, discharge_kw: float) -> None:
        """Update the state of charge given charge and discharge powers.

        Charging and discharging are applied with their respective
        efficiencies.  The resulting SoC is clamped to
        ``[soc_min, soc_max]``.
        """
        charge_kw = max(charge_kw, 0.0)
        discharge_kw = max(discharge_kw, 0.0)
        d_soc_ch = (charge_kw * self.dt_hours * self.batt.eta_ch) / self.batt.capacity_kwh
        d_soc_dis = (
            (discharge_kw * self.dt_hours) / max(self.batt.eta_dis, 1e-9)
        ) / self.batt.capacity_kwh
        soc_new = self._soc + d_soc_ch - d_soc_dis
        soc_new = float(
            np.clip(soc_new, self.batt.soc_min, self.batt.soc_max)
        )
        self._soc = soc_new

    # -----------------------------
    # Miscellaneous helpers
    # -----------------------------
    def _action_to_scalar(self, action: Any) -> float:
        """Convert an action from the environment's action space to a scalar.

        For continuous actions, this returns the first element of the
        array.  For discrete actions, it maps the integer index to
        evenly spaced values in [-1, 1].
        """
        if self.action_type == "continuous":
            a = np.asarray(action, dtype=np.float32).reshape(-1)[0]
            return float(a)
        # discrete
        idx = int(action)
        idx = int(np.clip(idx, 0, self.discrete_bins - 1))
        vals = np.linspace(-1.0, 1.0, self.discrete_bins, dtype=np.float32)
        return float(vals[idx])

    def _sample_start_index(self) -> int:
        """Sample or return the episode start index."""
        n = len(self.df)
        max_start = n - (self.episode_len_steps + self.horizon_steps + 1)
        if self.start_index is not None:
            return int(self.start_index)
        if self.random_start:
            return int(self.rng.integers(0, max_start + 1))
        return 0

    def _sample_init_soc(self) -> float:
        """Sample or return the initial state of charge."""
        if isinstance(self.init_soc, (float, int)):
            return float(np.clip(float(self.init_soc), self.batt.soc_min, self.batt.soc_max))
        if isinstance(self.init_soc, str) and self.init_soc.lower() == "random":
            lo, hi = (
                self.init_soc_range if self.init_soc_range is not None else (self.batt.soc_min, self.batt.soc_max)
            )
            lo = float(np.clip(lo, self.batt.soc_min, self.batt.soc_max))
            hi = float(np.clip(hi, self.batt.soc_min, self.batt.soc_max))
            if hi < lo:
                lo, hi = hi, lo
            return float(self.rng.uniform(lo, hi))
        raise ValueError("init_soc must be a float or 'random'.")

    def get_logs_df(self) -> pd.DataFrame:
        """Return a DataFrame of logs from the most recent episode."""
        return pd.DataFrame(self.logs)

    # Expose current episode indices (optional helpers)
    @property
    def episode_start_index(self) -> int:
        return self._t0

    @property
    def t(self) -> int:
        return self._t


# -----------------------------
# Minimal self-test / example
# -----------------------------
def _quick_rollout_example() -> None:
    """Run a short rollout with random actions for sanity checking.

    This function generates a synthetic dataset with sinusoidal price,
    load, and PV signals, instantiates the environment, and steps
    through 50 steps with random actions.  It prints the total reward
    and observation dimensionality.  The primary purpose is to ensure
    that the environment can be created and interacted with without
    raising exceptions.
    """
    # Create a synthetic dataset with 2000 rows of half-hour data
    n = 2000
    dt_index = pd.date_range("2022-01-01", periods=n, freq="30min")
    df = pd.DataFrame(
        {
            "dt": dt_index,
            "price": 5.0 + 2.0 * np.sin(np.linspace(0, 20, n)),
            "load": 1.5 + 0.5 * np.sin(np.linspace(0, 50, n)) + 0.2 * np.random.rand(n),
            "pv": np.maximum(0.0, 2.0 * np.sin(np.linspace(-3, 20, n))),
        }
    )
    env = HEMSEnv(
        df=df,
        episode_len_steps=336,
        forecast_cfg=ForecastConfig(horizon_hours=4, include_pv=True, include_load=True, style="interval", interval_pct=0.1),
        forecast_err_cfg=ForecastErrorConfig(enabled=True, error_types=("noise", "drift"), pv_error_pct=0.1, load_error_pct=0.1),
        time_feature_cfg=TimeFeatureConfig(include_doy=True, include_tod=True, include_dow=True),
        action_type="discrete",
        discrete_bins=11,
        init_soc="random",
        seed=42,
    )
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, inf = env.step(a)
        total_reward += r
        if term or trunc:
            break
    print("rollout ok, total_reward(50 steps) =", total_reward)
    print("obs_dim =", obs.shape[0], "logs_len =", len(env.logs))


if __name__ == "__main__":
    _quick_rollout_example()

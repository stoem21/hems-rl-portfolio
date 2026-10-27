"""Unified HEMS evaluation script.

Supports point, interval, and none (no-forecast) variants.
Replaces test_samp_point.py, test_samp_interval.py, and test_samp_no_fct.py.

Workers are pooled per checkpoint: policy weights and customer DataFrames are
loaded once per process, not once per task.

Examples
--------
# Point forecast agents
python test.py --forecast_style point \\
    --checkpoint_dir checkpoints_point/ --data_path test_dataset.csv \\
    --clusters test_clus_samp.json

# Interval forecast agents
python test.py --forecast_style interval \\
    --checkpoint_dir checkpoints_interval/ --data_path test_dataset.csv \\
    --clusters test_clus_samp.json

# No-forecast agents  (error injection is automatically disabled)
python test.py --forecast_style none \\
    --checkpoint_dir checkpoints_no_fct/ --data_path test_dataset.csv \\
    --clusters test_clus_samp.json
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from tianshou.data import Collector
from hems_env import HEMSEnv, BatterySpec, ForecastConfig, ForecastErrorConfig, TimeFeatureConfig
from train_helper import build_policy


# ─────────────────────────── dataclasses ───────────────────────────

@dataclass(frozen=True)
class ErrorVariant:
    tag: str            # "noise", "bias_p", "bias_n", "drift_p", "drift_n"
    error_type: str     # "noise" | "bias" | "drift"
    bias_sign: int = +1
    drift_sign: int = +1


@dataclass(frozen=True)
class Checkpoint:
    tag: str
    path: str
    horizon_hours: int  # 0 for none style
    style: str          # "point" | "interval" | "none"
    alg: str            # "TD3" | "SAC" | "DDPG"


@dataclass(frozen=True)
class TaskArgs:
    err: ErrorVariant
    size: float         # error percentage, e.g. 5.0 = 5%
    cust_id: int


# ─────────────────────── worker globals & init ─────────────────────

_W: Dict = {}


def _init_worker(cust_dfs: Dict[int, pd.DataFrame],
                 state_dict: dict,
                 alg: str,
                 horizon_hours: int,
                 style: str,
                 device: str,
                 force_len=None):
    _W["cust_dfs"]     = cust_dfs
    _W["alg"]          = alg
    _W["horizon_hours"] = horizon_hours
    _W["style"]        = style
    _W["device"]       = device
    # Optional per-customer episode-length override {cust_id: n_steps}. Lets a
    # caller score, e.g., no-forecast agents over the SAME horizon as forecast
    # agents. None -> each env keeps its natural length. The full df is still
    # used (so price_mean etc. are unchanged); only the episode is cut shorter.
    _W["force_len"]    = force_len

    first_cid = next(iter(cust_dfs))
    dummy_env = _build_env(cust_dfs[first_cid], _base_cfg(horizon_hours, style), first_cid)
    policy, _ = build_policy(dummy_env, alg, device=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    _W["policy"] = policy


# ─────────────────────────── env builder ───────────────────────────

_BATTERY = BatterySpec(
    capacity_kwh=13.5, p_ch_max_kw=4.6, p_dis_max_kw=4.6,
    eta_ch=1.0, eta_dis=1.0, soc_min=0.10, soc_max=0.90,
)
_TFEAT = TimeFeatureConfig(include_doy=False, include_tod=True, include_dow=True)


def _base_cfg(horizon_hours: int, style: str) -> dict:
    """Minimal cfg used when building the dummy env for policy construction."""
    return {
        "horizon_hours": horizon_hours,
        "style": style,
        "error": False,
        "error_type": "noise",
        "error_size": 0.0,
        "bias_sign": +1,
        "drift_sign": +1,
    }


def _build_env(df_use: pd.DataFrame, cfg: dict, cust_id: int) -> HEMSEnv:
    is_none = cfg["style"] == "none"

    if is_none:
        forecast_cfg = ForecastConfig(
            enabled=False,
            horizon_hours=0,
            include_pv=False,
            include_load=False,
            include_price=False,
            style="point",
            interval_pct=0,
        )
        ferr = ForecastErrorConfig(enabled=False)
        episode_len = len(df_use) - 1
    else:
        forecast_cfg = ForecastConfig(
            enabled=True,
            horizon_hours=cfg["horizon_hours"],
            include_pv=True,
            include_load=True,
            include_price=True,
            style=cfg["style"],
            interval_pct=0.10,
        )
        ferr = ForecastErrorConfig(
            enabled=cfg["error"],
            random=False,
            error_types=(cfg["error_type"],),
            pv_error_pct=cfg["error_size"],
            load_error_pct=cfg["error_size"],
            bias_sign=cfg["bias_sign"],
            drift_sign=cfg["drift_sign"],
            noise_clip=0.50,
        )
        episode_len = len(df_use) - (cfg["horizon_hours"] * 2 + 1)

    return HEMSEnv(
        df=df_use,
        episode_len_steps=episode_len,
        action_type="continuous",
        discrete_bins=3,
        allow_grid_charge=True,
        random_start=False,
        start_index=0,
        init_soc=0,
        log_enabled=False,
        battery_spec=_BATTERY,
        forecast_cfg=forecast_cfg,
        forecast_err_cfg=ferr,
        time_feature_cfg=_TFEAT,
        customer_id=cust_id,
        reward_mode="cost",
        seed=42,
    )


# ─────────────────────────── worker task ───────────────────────────

def task(task_args: TaskArgs) -> dict:
    is_none = _W["style"] == "none"
    cfg = {
        "horizon_hours": 0 if is_none else _W["horizon_hours"],
        "style": _W["style"],
        "error": False if is_none else True,
        "error_type": task_args.err.error_type,
        "error_size": task_args.size / 100,
        "bias_sign": task_args.err.bias_sign,
        "drift_sign": task_args.err.drift_sign,
    }
    env = _build_env(_W["cust_dfs"][task_args.cust_id], cfg, task_args.cust_id)
    force_len = _W.get("force_len")
    if force_len is not None:
        n = force_len[task_args.cust_id] if isinstance(force_len, dict) else force_len
        env.episode_len_steps = int(n)
    collector = Collector(_W["policy"], env)
    with torch.no_grad():
        result = collector.collect(n_episode=1, reset_before_collect=True)
    return {"result": result, "customer_id": task_args.cust_id}


# ─────────────────────────── helpers ───────────────────────────────

def _discover_checkpoints(checkpoint_dir: str, style: str) -> list[Checkpoint]:
    """Scan a checkpoint directory for best.pt files and build Checkpoint objects."""
    chkp_list = []
    p = Path(checkpoint_dir)
    for folder in sorted(p.iterdir()):
        if not folder.is_dir():
            continue
        best = folder / "best.pt"
        if not best.exists():
            continue
        name = folder.name
        if "DDPG" in name:
            alg = "DDPG"
        elif "SAC" in name:
            alg = "SAC"
        elif "TD3" in name:
            alg = "TD3"
        else:
            print(f"  SKIP {name}: cannot infer algorithm")
            continue
        # infer horizon from folder name (e.g. "…_point16" → 16); default 16
        horizon = 0
        if style != "none":
            for part in name.split("_"):
                digits = "".join(c for c in part if c.isdigit())
                if digits:
                    horizon = int(digits)
        chkp_list.append(Checkpoint(name, str(best), horizon, style, alg))
    return chkp_list


def _load_from_mlflow(style: str, alg: str, seed: int):
    """Try to load a checkpoint from MLflow registry (seed-tagged). Returns None on failure."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        # Map style → variant prefix for model name
        prefix_map = {"interval": "I", "point": "N", "none": "R"}
        prefix = prefix_map.get(style, style)
        model_name = f"hems-{prefix}-{alg}"
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{model_name}' and tags.seed='{seed}'"
        )
        if versions:
            best_v = max(versions, key=lambda v: int(v.version))
            uri = f"models:/{model_name}/{best_v.version}"
        else:
            uri = f"models:/{model_name}/latest"
        local_path = mlflow.artifacts.download_artifacts(uri)
        ckpt_file = os.path.join(local_path, "best.pt") if os.path.isdir(local_path) else local_path
        if os.path.exists(ckpt_file):
            return torch.load(ckpt_file, map_location="cpu")
    except Exception as e:
        print(f"  MLflow load failed ({e}), falling back to local file")
    return None


# ─────────────────────────── main ──────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate HEMS agents across error variants and sizes")
    ap.add_argument("--forecast_style", required=True, choices=["point", "interval", "none"])
    ap.add_argument("--checkpoint_dir", required=True,
                    help="Directory containing run subdirs each with a best.pt")
    ap.add_argument("--data_path",      required=True, help="Test dataset CSV")
    ap.add_argument("--clusters",       required=True,
                    help="JSON mapping cluster_id -> list of customer_id")
    ap.add_argument("--output_dir",     default=None,
                    help="Root dir for CSV results (default: log_<style>/)")
    ap.add_argument("--data_offset",    type=int, default=35088,
                    help="Row offset into each customer's slice (default: 35088 ≈ 2 years of 30-min data)")
    ap.add_argument("--seed",           type=int, default=None,
                    help="If set, attempt to load matching seed from MLflow registry")
    ap.add_argument("--device",         default="cpu")
    args = ap.parse_args()

    style_label = "no_fct" if args.forecast_style == "none" else args.forecast_style
    out_root = args.output_dir or f"log_{style_label}"

    # ── 1. Load customer list ──────────────────────────────────────
    with open(args.clusters, "rb") as f:
        cluster_data = json.load(f)
    cust_list = [x for v in cluster_data.values() for x in v]

    # ── 2. Pre-filter customer DataFrames once ─────────────────────
    print("Loading & pre-filtering test data …")
    raw = pd.read_csv(args.data_path)
    raw = raw.drop(raw.columns[0], axis=1)
    cust_dfs: Dict[int, pd.DataFrame] = {}
    for cid in cust_list:
        cdf = raw[raw["customer_id"] == cid]
        cust_dfs[cid] = cdf.iloc[args.data_offset:].reset_index(drop=True)
    del raw

    # ── 3. Discover checkpoints ────────────────────────────────────
    chkp_list = _discover_checkpoints(args.checkpoint_dir, args.forecast_style)
    if not chkp_list:
        print(f"No valid checkpoints found in {args.checkpoint_dir}")
        return
    print(f"Found {len(chkp_list)} checkpoint(s)")

    # ── 4. Error variants and sizes ───────────────────────────────
    if args.forecast_style == "none":
        # No forecast → no error injection
        errors = [ErrorVariant("none", "noise")]
        sizes  = [0.0]
    else:
        errors = [
            ErrorVariant("noise",   "noise", +1, +1),
            ErrorVariant("bias_p",  "bias",  +1, +1),
            ErrorVariant("bias_n",  "bias",  -1, +1),
            ErrorVariant("drift_p", "drift", +1, +1),
            ErrorVariant("drift_n", "drift", +1, -1),
        ]
        sizes = [0.0, 5.0, 10.0, 15.0, 20.0]

    # ── 5. Evaluate: one Pool per checkpoint ───────────────────────
    for chkp in chkp_list:
        print(f"\n{'='*55}")
        print(f"Checkpoint: {chkp.tag}  |  alg={chkp.alg}  style={chkp.style}")
        print(f"{'='*55}")

        # Try MLflow registry first if seed is given; fall back to local file
        loaded = None
        if args.seed is not None:
            loaded = _load_from_mlflow(chkp.style, chkp.alg, args.seed)
        if loaded is None:
            loaded = torch.load(chkp.path, map_location="cpu")
        state_dict = loaded["policy"] if isinstance(loaded, dict) and "policy" in loaded else loaded

        with Pool(
            initializer=_init_worker,
            initargs=(cust_dfs, state_dict, chkp.alg,
                      chkp.horizon_hours, chkp.style, args.device),
        ) as pool:
            for err in errors:
                print(f"  error: {err.tag}")
                for size in sizes:
                    if size == 0 and err.tag not in ("noise", "none"):
                        continue

                    task_list = [TaskArgs(err, size, cid) for cid in cust_list]
                    results = pool.map(task, task_list)

                    out_dir = os.path.join(out_root, chkp.tag, err.tag)
                    os.makedirs(out_dir, exist_ok=True)
                    csv_path = os.path.join(out_dir, f"{size}.csv")
                    total = 0.0
                    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(["customer_id", "cost"])
                        for r in results:
                            cost = r["result"].returns_stat.mean
                            writer.writerow([r["customer_id"], cost])
                            total += cost
                    print(f"    size={size}  mean_cost={total / len(task_list):.4f}  -> {csv_path}")


if __name__ == "__main__":
    main()

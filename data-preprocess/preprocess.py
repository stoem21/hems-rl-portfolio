"""Ausgrid → full_dataset.csv preprocessor.

Converts the three raw Ausgrid Solar Home Electricity Data CSV files into
the normalised long-format CSV expected by the HEMS training pipeline.

Output columns
--------------
customer_id : int     Household identifier (1–300)
dt          : str     Interval start timestamp  YYYY-MM-DD HH:MM:SS (30-min steps)
load        : float   Household electricity demand  (kW, GC + CL categories summed)
pv          : float   Rooftop solar gross generation (kW, GG category)
price       : float   Time-of-use import tariff     (AUD/kWh, 3-tier fixed schedule)

Usage
-----
    python preprocess.py                      # uses DEFAULT_PATHS / DEFAULT_OUTPUT
    python preprocess.py --help
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ausgrid file paths (edit to match your download location)
# ---------------------------------------------------------------------------
DEFAULT_PATHS = [
    "ausgrid/2010-2011 Solar home electricity data.csv",
    "ausgrid/2011-2012 Solar home electricity data v2.csv",
    "ausgrid/2012-2013 Solar home electricity data v2.csv",
]
DEFAULT_OUTPUT = "../data/full_dataset.csv"

# ---------------------------------------------------------------------------
# TOU tariff schedule (NSW residential, 3-tier)
# ---------------------------------------------------------------------------
# Off-peak  : 10:30–14:00  (8 half-hour slots starting at those times)
# Peak      : 15:30–21:00  (12 half-hour slots)
# Shoulder  : all other times
_PRICE_OFF_PEAK = 0.2332   # AUD/kWh
_PRICE_SHOULDER = 0.3276
_PRICE_PEAK     = 0.4957


def tou_price(hour: int, minute: int) -> float:
    """Return the TOU import tariff for the 30-min interval starting at hour:minute."""
    slot = hour * 2 + minute // 30   # 0–47
    if 21 <= slot <= 28:             # 10:30–14:00
        return _PRICE_OFF_PEAK
    if 31 <= slot <= 42:             # 15:30–21:00
        return _PRICE_PEAK
    return _PRICE_SHOULDER


# ---------------------------------------------------------------------------
# Raw CSV helpers
# ---------------------------------------------------------------------------
def _detect_date_col(columns: list[str]) -> str | None:
    cols = [c.strip() for c in columns]
    for c in cols:
        if c.lower() == "date":
            return c
    for c in cols:
        if "date" in c.lower():
            return c
    return None


def _infer_date_format(sample: str) -> str | None:
    s = str(sample).strip()
    if re.fullmatch(r"\d{1,2}-[A-Za-z]{3}-\d{2}", s):
        return "%d-%b-%y"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
        return "%d/%m/%Y"
    return None


def _time_col_to_slot_start(col: str) -> tuple[int, int]:
    """Return (hour, minute) for the START of the 30-min interval.

    Ausgrid column names are the END time of each 30-min interval, so the
    start is 30 minutes earlier.  The special column '0:00' represents the
    last slot of the day (23:30–00:00), whose start is 23:30.
    """
    col = col.strip()
    if col == "0:00":
        return 23, 30
    h, m = col.split(":")
    total = int(h) * 60 + int(m) - 30
    return total // 60, total % 60


def _sort_key(col: str) -> int:
    col = col.strip()
    if col == "0:00":
        return 24 * 60
    h, m = col.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------
# Core transformer
# ---------------------------------------------------------------------------
def build_full_dataset(raw_paths: list[str | Path], output_path: str | Path | None = None) -> pd.DataFrame:
    """Convert raw Ausgrid CSV files to the normalised full_dataset format.

    Parameters
    ----------
    raw_paths:
        Paths to the three annual Ausgrid CSVs (order does not matter — they
        are concatenated and de-duplicated by customer × dt).
    output_path:
        If given, the resulting DataFrame is written to this path as CSV.

    Returns
    -------
    pd.DataFrame with columns: customer_id, dt, load, pv, price
    """
    load_frames: list[pd.DataFrame] = []
    pv_frames:   list[pd.DataFrame] = []

    for p in raw_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Raw CSV not found: {p}")

        hdr = pd.read_csv(p, nrows=0)
        cols = [c.strip() for c in hdr.columns.astype(str)]
        date_col = _detect_date_col(cols)
        if date_col is None:
            raise ValueError(f"Could not detect a date column in {p.name}")

        time_cols = sorted(
            [c for c in cols if re.fullmatch(r"\d{1,2}:\d{2}", c)],
            key=_sort_key,
        )
        if not time_cols:
            raise ValueError(f"No 30-min time columns found in {p.name}")

        usecols = [c for c in ["Customer", "Consumption Category", date_col] if c in cols] + time_cols
        df = pd.read_csv(p, usecols=usecols)
        df.columns = [c.strip() for c in df.columns]

        sample = df[date_col].dropna().astype(str).iloc[0]
        fmt = _infer_date_format(sample)
        df["_date"] = (
            pd.to_datetime(df[date_col], format=fmt, errors="coerce")
            if fmt
            else pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        )
        df = df.dropna(subset=["Customer", "_date"])
        df["Customer"] = df["Customer"].astype(int)
        df["Consumption Category"] = df["Consumption Category"].astype(str).str.upper().str.strip()
        df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # --- Load: sum GC + CL per customer × date ---
        load_df = (
            df[df["Consumption Category"].isin({"GC", "CL"})]
            .groupby(["Customer", "_date"], sort=False)[time_cols]
            .sum()
            .reset_index()
        )
        load_frames.append(load_df)

        # --- PV: GG rows (gross solar generation) ---
        pv_df = (
            df[df["Consumption Category"] == "GG"]
            .groupby(["Customer", "_date"], sort=False)[time_cols]
            .max()   # one GG row per customer×date; max handles duplicates safely
            .reset_index()
        )
        pv_frames.append(pv_df)

    load_wide = pd.concat(load_frames, ignore_index=True).drop_duplicates(["Customer", "_date"])
    pv_wide   = pd.concat(pv_frames,  ignore_index=True).drop_duplicates(["Customer", "_date"])

    # Build column → (hour, minute) map once
    col_to_hm = {c: _time_col_to_slot_start(c) for c in time_cols}

    def _melt_to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
        long = wide.melt(
            id_vars=["Customer", "_date"],
            value_vars=time_cols,
            var_name="_time_col",
            value_name=value_name,
        )
        long["_h"] = long["_time_col"].map(lambda c: col_to_hm[c][0])
        long["_m"] = long["_time_col"].map(lambda c: col_to_hm[c][1])
        long["dt"] = long["_date"] + pd.to_timedelta(long["_h"] * 60 + long["_m"], unit="min")
        return long[["Customer", "dt", value_name]].rename(columns={"Customer": "customer_id"})

    load_long = _melt_to_long(load_wide, "load")
    pv_long   = _melt_to_long(pv_wide,   "pv")

    df_out = load_long.merge(pv_long, on=["customer_id", "dt"], how="left")
    df_out["pv"] = df_out["pv"].fillna(0.0)
    df_out["price"] = df_out["dt"].apply(lambda t: tou_price(t.hour, t.minute))

    df_out = df_out.sort_values(["customer_id", "dt"]).reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False)
        print(f"Wrote {len(df_out):,} rows → {out}")

    return df_out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--raw",
        nargs="+",
        default=DEFAULT_PATHS,
        metavar="CSV",
        help="Paths to the raw Ausgrid CSV files (default: %(default)s)",
    )
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Output CSV path (default: %(default)s)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        build_full_dataset(args.raw, args.output)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

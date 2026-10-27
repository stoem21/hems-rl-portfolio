"""gen_dashboard_data.py — regenerate the dashboard's baked data file.

Reads the completed evaluation sweep, the household->cluster map, and the raw
dataset (for the no-battery baseline), and writes
hems-frontend/src/data/dashboardData.js covering all evaluated households.

Sources
  eval_results/test_households_sweep.csv   long-format sweep (agents + RULE)
  data-preprocess/test_clus_full.json      {household: cluster} for all households
  hems-backend/data/full_dataset.csv       raw load/pv/price (no-battery baseline)

Run from the repo root:  python gen_dashboard_data.py
"""

import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(REPO_ROOT, "hems-backend")
for _p in (BACKEND, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_config import (                                            # noqa: E402
    hh_role, hh_cluster, TRAIN_HOUSEHOLDS, TEST_HOUSEHOLDS, TRAIN_FLAT, TEST_FLAT,
)

OUT_CSV = os.path.join(REPO_ROOT, "eval_results", "test_households_sweep.csv")
CLUS_JSON = os.path.join(REPO_ROOT, "data-preprocess", "test_clus_full.json")
DATA_PATH = os.path.join(BACKEND, "data", "full_dataset.csv")
JS_OUT = os.path.join(REPO_ROOT, "private-results", "dashboardData.js")

TEST_START = "2012-07-01"
HORIZON_HOURS = 16
LOOKAHEAD = HORIZON_HOURS * 2 + 1
DT_HOURS = 0.5

# Agent order matches the current dashboard (R, V, N, I families).
AGENTS = ["R-DDPG", "R-SAC", "R-TD3",
          "V-DDPG", "V-SAC", "V-TD3",
          "N-DDPG", "N-SAC", "N-TD3",
          "I-DDPG", "I-SAC", "I-TD3"]
FORECAST_AGENTS = [a for a in AGENTS if a[0] in ("I", "N", "V")]
STRATEGIES = ["no_battery", "rule_based"] + AGENTS
ERROR_TYPES = ["noise", "drift_p", "drift_n", "bias_p", "bias_n"]
ERROR_PCTS = ["5.0", "10.0", "15.0", "20.0"]

# Static constants carried over verbatim from the current dashboardData.js.
STATIC_JS = """export const STRATEGY_COLORS = {
  'no_battery': '#D85A30',
  'rule_based': '#D85A30',
  'R-DDPG': '#888780', 'R-SAC': '#888780', 'R-TD3': '#888780',
  'V-DDPG': '#378ADD', 'V-SAC': '#378ADD', 'V-TD3': '#378ADD',
  'N-DDPG': '#1D9E75', 'N-SAC': '#1D9E75', 'N-TD3': '#1D9E75',
  'I-DDPG': '#639922', 'I-SAC': '#639922', 'I-TD3': '#639922',
}

export const STRATEGY_LABELS = {
  'no_battery': 'No battery', 'rule_based': 'Rule-based',
  'R-DDPG': 'Reactive', 'R-SAC': 'Reactive', 'R-TD3': 'Reactive',
  'V-DDPG': 'Vanilla', 'V-SAC': 'Vanilla', 'V-TD3': 'Vanilla',
  'N-DDPG': 'Noise', 'N-SAC': 'Noise', 'N-TD3': 'Noise',
  'I-DDPG': 'Interval', 'I-SAC': 'Interval', 'I-TD3': 'Interval',
}

export const ALL_ALGOS = ['no_battery','rule_based','R-DDPG','R-SAC','R-TD3','V-DDPG','V-SAC','V-TD3','N-DDPG','N-SAC','N-TD3','I-DDPG','I-SAC','I-TD3']
// ERR_ALGOS: agents that support forecast-error injection (used by RobustnessHeatmap)
export const ERR_ALGOS = ['V-DDPG','V-SAC','V-TD3','N-DDPG','N-SAC','N-TD3','I-DDPG','I-SAC','I-TD3']
// SIM_ALGOS: all agents available for live simulation, including no-forecast R-* agents
export const SIM_ALGOS = ['R-TD3','R-SAC','R-DDPG','V-DDPG','V-SAC','V-TD3','N-DDPG','N-SAC','N-TD3','I-DDPG','I-SAC','I-TD3']
export const ERROR_TYPES = ['noise','drift_p','drift_n','bias_p','bias_n']
export const ERROR_PCTS = ['5.0','10.0','15.0','20.0']

// Time split constants (Ausgrid dataset)
export const DATE_DATASET_START = '2010-07-01'
export const DATE_TRAIN_END     = '2012-01-01'
export const DATE_VAL_END       = '2012-07-01'
export const DATE_TEST_START    = '2012-07-01'
export const SEEDS = [1, 2, 3, 4, 5]
"""


def r2(x):
    return round(float(x), 2)


def compute_no_battery(households):
    """Do-nothing baseline per household: pay for net grid import (no battery),
    summed over the same forecast-length episode window the agents use."""
    print("computing no-battery baseline (loading dataset)...")
    raw = pd.read_csv(DATA_PATH, usecols=["customer_id", "dt", "load", "pv", "price"],
                      parse_dates=["dt"])
    raw["customer_id"] = pd.to_numeric(raw["customer_id"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["customer_id"]).copy()
    raw["customer_id"] = raw["customer_id"].astype(int)
    ts = pd.Timestamp(TEST_START)
    out = {}
    for cid in households:
        cdf = raw[raw["customer_id"] == cid].sort_values("dt").reset_index(drop=True)
        if cdf.empty:
            continue
        pos = cdf["dt"].searchsorted(ts, side="left")
        cdf = cdf.iloc[pos:].reset_index(drop=True)
        n = len(cdf) - LOOKAHEAD
        if n <= 0:
            continue
        w = cdf.iloc[:n]
        net = (w["load"] - w["pv"]).clip(lower=0)
        out[cid] = r2((net * w["price"] * DT_HOURS).sum())
    return out


def main():
    clus = {int(k): int(v) for k, v in json.load(open(CLUS_JSON)).items()}

    res = pd.read_csv(OUT_CSV)
    # Defensive dedup: the CSV accumulated a few duplicate rows over multiple runs.
    before = len(res)
    res = res.drop_duplicates(
        subset=["agent", "seed", "customer_id", "error_tag", "error_size"], keep="last")
    if len(res) != before:
        print(f"deduped {before - len(res)} duplicate rows")

    households = sorted(int(c) for c in res["customer_id"].unique())
    print(f"{len(households)} households, {len(res)} rows")

    no_batt = compute_no_battery(households)

    # ── clean cost per (agent, seed, household): the error_size==0 rows ──
    clean = res[res["error_size"] == 0.0]
    g = (clean.groupby(["agent", "seed", "customer_id"])["cost"].mean().reset_index())
    # seed-averaged per household per agent
    hh_agent = (g.groupby(["customer_id", "agent"])["cost"].mean())

    def strat_cost_hh(hh, strat):
        if strat == "no_battery":
            return no_batt.get(hh)
        key = "RULE" if strat == "rule_based" else strat
        try:
            return r2(hh_agent.loc[(hh, key)])
        except KeyError:
            return None

    # ── HH_DATA ──
    HH_DATA = {}
    for hh in households:
        costs = {}
        for s in STRATEGIES:
            v = strat_cost_hh(hh, s)
            if v is not None:
                costs[s] = v
        HH_DATA[str(hh)] = {"cluster": clus.get(hh), "costs": costs}

    # ── ALGO_MEANS / SAVING_PCT / NO_ERR_MEANS (mean over households) ──
    def mean_over_hh(strat):
        vals = [HH_DATA[str(hh)]["costs"].get(strat) for hh in households]
        vals = [v for v in vals if v is not None]
        return r2(sum(vals) / len(vals)) if vals else None

    ALGO_MEANS = {s: mean_over_hh(s) for s in STRATEGIES}
    nb_mean = ALGO_MEANS["no_battery"]
    SAVING_PCT = {s: (r2((nb_mean - ALGO_MEANS[s]) / nb_mean * 100) if nb_mean else 0.0)
                  for s in STRATEGIES}
    NO_ERR_MEANS = {a: ALGO_MEANS[a] for a in AGENTS}

    # ── CLUSTER_DATA (mean over households in each cluster) ──
    CLUSTER_DATA = {}
    for c in sorted(set(clus[hh] for hh in households)):
        members = [hh for hh in households if clus.get(hh) == c]
        cd = {}
        for s in STRATEGIES:
            vals = [HH_DATA[str(hh)]["costs"].get(s) for hh in members]
            vals = [v for v in vals if v is not None]
            if vals:
                cd[s] = r2(sum(vals) / len(vals))
        CLUSTER_DATA[str(c)] = cd

    # ── SEED_HH_DATA[seed][hh][strategy] ──
    seeds = sorted(int(s) for s in g["seed"].unique() if int(s) != 0)
    per = g.set_index(["seed", "customer_id", "agent"])["cost"]
    SEED_HH_DATA = {}
    for seed in seeds:
        sd = {}
        for hh in households:
            row = {}
            if no_batt.get(hh) is not None:
                row["no_battery"] = no_batt[hh]
            # rule_based is seed-independent (seed 0) -> repeat
            rb = strat_cost_hh(hh, "rule_based")
            if rb is not None:
                row["rule_based"] = rb
            for a in AGENTS:
                try:
                    row[a] = r2(per.loc[(seed, hh, a)])
                except KeyError:
                    pass
            sd[str(hh)] = row
        SEED_HH_DATA[str(seed)] = sd

    # ── ROBUSTNESS (forecast agents, error rows) ──
    err = res[(res["error_size"] > 0) & (res["agent"].isin(FORECAST_AGENTS))]
    rob = err.groupby(["agent", "error_tag", "error_size"])["cost"].mean()
    ROBUSTNESS = {}
    for a in FORECAST_AGENTS:
        et = {}
        for e in ERROR_TYPES:
            pcts = {}
            for p in ERROR_PCTS:
                try:
                    pcts[p] = r2(rob.loc[(a, e, float(p))])
                except KeyError:
                    pass
            if pcts:
                et[e] = pcts
        ROBUSTNESS[a] = et

    robhh = err.groupby(["customer_id", "error_tag", "error_size", "agent"])["cost"].mean()
    ROBUSTNESS_BY_HH = {}
    for hh in households:
        et = {}
        for e in ERROR_TYPES:
            pcts = {}
            for p in ERROR_PCTS:
                ag = {}
                for a in FORECAST_AGENTS:
                    try:
                        ag[a] = r2(robhh.loc[(hh, e, float(p), a)])
                    except KeyError:
                        pass
                if ag:
                    pcts[p] = ag
            if pcts:
                et[e] = pcts
        if et:
            ROBUSTNESS_BY_HH[str(hh)] = et

    # ── metadata ──
    # HH_META must cover every household the UI references: the evaluated set
    # (1..100) plus the train/test households from data_config (whose ids can be
    # outside 1..100). Cluster comes from test_clus_full.json where available,
    # else from data_config's train/test cluster mapping.
    def cluster_of(hh):
        return clus[hh] if hh in clus else hh_cluster(hh)

    meta_ids = sorted(set(households) | set(TRAIN_FLAT) | set(TEST_FLAT))
    HH_META = {str(hh): {"role": hh_role(hh), "cluster": cluster_of(hh)} for hh in meta_ids}

    # ── emit ──
    def js(name, obj, comment=None):
        line = f"export const {name} = {json.dumps(obj, separators=(',', ':'))}\n"
        return (f"// {comment}\n" + line) if comment else line

    parts = [
        "// AUTO-GENERATED by gen_dashboard_data.py — do not edit by hand.\n"
        "// Source: eval_results/test_households_sweep.csv (100-household RL evaluation).\n\n",
        js("ALGO_MEANS", ALGO_MEANS),
        js("SAVING_PCT", SAVING_PCT),
        js("HH_DATA", HH_DATA),
        js("CLUSTER_DATA", CLUSTER_DATA),
        js("ROBUSTNESS", ROBUSTNESS),
        js("NO_ERR_MEANS", NO_ERR_MEANS),
        "\n" + STATIC_JS + "\n",
        js("HOUSEHOLDS", households),
        js("CLUSTERS", sorted(set(clus[hh] for hh in households))),
        js("TRAIN_HOUSEHOLDS", {str(k): v for k, v in TRAIN_HOUSEHOLDS.items()}),
        js("TEST_HOUSEHOLDS", {str(k): v for k, v in TEST_HOUSEHOLDS.items()}),
        js("ALL_HOUSEHOLDS", households),
        js("HH_META", HH_META),
        js("SEED_HH_DATA", SEED_HH_DATA,
           "SEED_HH_DATA[seed][householdId][agent] = cost (AUD/yr)"),
        js("ROBUSTNESS_BY_HH", ROBUSTNESS_BY_HH,
           "ROBUSTNESS_BY_HH[householdId][error_type][pct][agent] = cost, seed-averaged"),
    ]

    os.makedirs(os.path.dirname(JS_OUT), exist_ok=True)
    with open(JS_OUT, "w") as f:
        f.write("".join(parts))

    print(f"wrote {JS_OUT}")
    print(f"  households={len(households)} clusters={sorted(set(clus[hh] for hh in households))}")
    print("  ALGO_MEANS:", {k: ALGO_MEANS[k] for k in ("no_battery", "rule_based",
                                                        "R-TD3", "V-TD3", "N-TD3", "I-TD3")})
    print("  SAVING_PCT I-TD3:", SAVING_PCT["I-TD3"], "| rule_based:", SAVING_PCT["rule_based"])


if __name__ == "__main__":
    main()

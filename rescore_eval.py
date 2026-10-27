"""rescore_eval.py — complete eval_results/test_households_sweep.csv.

Two steps, mirroring notebook cells 5c and 5d:

  5c  Add the rule-based baseline as a pseudo-agent "RULE" (one row per household,
      TOU + solar heuristic, no forecasts), scored at the forecast episode length.
  5d  Re-score every no-forecast controller (R-* agents + RULE) at the SAME episode
      length as the forecast agents (len - LOOKAHEAD), so all costs cover the same
      horizon. Overwrites those rows in place; forecast-agent rows are untouched.

Idempotent: RULE rows are only added where missing; re-scoring recomputes style in
{none, rule}. A timestamped .bak copy of the CSV is written before any change.

Run from the repo root:  python rescore_eval.py
"""

import csv
import os
import shutil
import sys
import time
from multiprocessing import Pool

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(REPO_ROOT, "hems-backend")
for _p in (BACKEND, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test import ErrorVariant, TaskArgs, _init_worker, task          # noqa: E402
from agent_loader import AGENT_ALG_MAP, build_sim_env                # noqa: E402
from data_config import hh_cluster                                   # noqa: E402

DATA_PATH = os.path.join(BACKEND, "data", "full_dataset.csv")
AGENTS_DIR = os.path.join(REPO_ROOT, "agent")
OUT_CSV = os.path.join(REPO_ROOT, "eval_results", "test_households_sweep.csv")

HOUSEHOLDS = list(range(1, 101))
TEST_START = "2012-07-01"
HORIZON_HOURS = 16
LOOKAHEAD = HORIZON_HOURS * 2 + 1           # forecast lookahead (steps)
DEVICE = "cpu"
N_WORKERS = min(8, os.cpu_count() or 1)

RULE_OFF_PEAK_MAX = 0.27
RULE_PEAK_MIN = 0.45
NONE_ERRORS = [ErrorVariant("none", "noise")]
FIELDS = ["agent", "seed", "alg", "style", "customer_id", "cluster",
          "error_tag", "error_size", "cost"]


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cust_dfs():
    raw = pd.read_csv(DATA_PATH, usecols=["customer_id", "dt", "load", "pv", "price"],
                      parse_dates=["dt"])
    raw["customer_id"] = pd.to_numeric(raw["customer_id"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["customer_id"]).copy()
    raw["customer_id"] = raw["customer_id"].astype(int)
    ts = pd.Timestamp(TEST_START)
    cust = {}
    for cid in HOUSEHOLDS:
        cdf = raw[raw["customer_id"] == cid].sort_values("dt").reset_index(drop=True)
        if cdf.empty:
            continue
        pos = cdf["dt"].searchsorted(ts, side="left")
        cdf = cdf.iloc[pos:].reset_index(drop=True)
        if len(cdf) <= LOOKAHEAD + 1:
            continue
        cust[cid] = cdf
    return cust


def rule_action(row):
    price = float(row["price"])
    load = float(row["load"])
    pv = float(row["pv"])
    if pv > load:
        return 1.0
    if price <= RULE_OFF_PEAK_MAX:
        return 1.0
    if price >= RULE_PEAK_MIN:
        return -1.0
    return 0.0


def rule_cost(cust_dfs, cid):
    df = cust_dfs[cid]
    env = build_sim_env(df, "R-TD3", cid, error_type="none",
                        episode_len_steps=len(df) - LOOKAHEAD)
    env.reset()
    total_reward = 0.0
    t = 0
    while True:
        _, rew, term, trunc, _ = env.step(rule_action(df.iloc[t]))
        total_reward += float(rew)
        t += 1
        if term or trunc:
            break
    return -total_reward


def main():
    if not os.path.exists(OUT_CSV):
        raise SystemExit(f"{OUT_CSV} not found — run the evaluation sweep first.")
    shutil.copy2(OUT_CSV, OUT_CSV + time.strftime(".%Y%m%d-%H%M%S.bak"))

    cust_dfs = load_cust_dfs()
    print(f"loaded {len(cust_dfs)} households")

    # ── 5c: add RULE rows where missing ──
    df = pd.read_csv(OUT_CSV)
    rule_have = set(df.loc[df["agent"] == "RULE", "customer_id"].astype(int))
    todo = [c for c in cust_dfs if c not in rule_have]
    print(f"5c RULE: {len(rule_have)} present, {len(todo)} to add")
    if todo:
        with open(OUT_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            for cid in todo:
                w.writerow(dict(agent="RULE", seed=0, alg="rule", style="rule",
                                customer_id=cid, cluster=hh_cluster(cid),
                                error_tag="none", error_size=0.0,
                                cost=rule_cost(cust_dfs, cid)))
        print(f"  added {len(todo)} RULE rows")

    # ── 5d: re-score none/rule at the forecast episode length ──
    res = pd.read_csv(OUT_CSV)
    forced = {cid: len(d) - LOOKAHEAD for cid, d in cust_dfs.items()}
    new_cost = {}

    r_ckpts = sorted({(a, int(s)) for a, s in
                      zip(res.loc[res["style"] == "none", "agent"],
                          res.loc[res["style"] == "none", "seed"])})
    print(f"5d re-score: {len(r_ckpts)} R-* checkpoints × {len(cust_dfs)} households")
    t0 = time.time()
    for a, s in r_ckpts:
        path = os.path.join(AGENTS_DIR, f"{a}-{s}.pt")
        if not os.path.exists(path):
            print(f"   MISSING {path}")
            continue
        loaded = _torch_load(path)
        sd = loaded["policy"] if isinstance(loaded, dict) and "policy" in loaded else loaded
        tl = [TaskArgs(NONE_ERRORS[0], 0.0, cid) for cid in cust_dfs]
        initargs = (cust_dfs, sd, AGENT_ALG_MAP[a], 0, "none", DEVICE, forced)
        if N_WORKERS == 1:
            _init_worker(*initargs)
            results = [task(t) for t in tl]
        else:
            with Pool(N_WORKERS, initializer=_init_worker, initargs=initargs) as pool:
                results = list(pool.imap(task, tl))
        for r in results:
            new_cost[(a, s, r["customer_id"])] = -float(r["result"].returns_stat.mean)
        print(f"   {a}-{s}: {len(results)} rescored (elapsed {(time.time()-t0)/60:.1f} min)")

    rule_cids = res.loc[res["style"] == "rule", "customer_id"].astype(int).unique()
    for cid in rule_cids:
        if int(cid) in cust_dfs:
            new_cost[("RULE", 0, int(cid))] = rule_cost(cust_dfs, int(cid))
    print(f"re-scored {len(r_ckpts)} R-* checkpoints and {len(rule_cids)} RULE rows")

    def _newcost(r):
        if r["style"] in ("none", "rule"):
            k = (r["agent"], int(r["seed"]), int(r["customer_id"]))
            if k in new_cost:
                return new_cost[k]
        return r["cost"]

    res["cost"] = res.apply(_newcost, axis=1)
    res.to_csv(OUT_CSV, index=False)
    print(f"done — updated {len(new_cost)} none/rule rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()

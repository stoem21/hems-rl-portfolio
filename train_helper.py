import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


# -----------------------------
# Utilities: data + pools
# -----------------------------
def load_full_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Ensure required columns exist
    req = {"customer_id", "dt", "load", "pv", "price"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Sort so each customer is contiguous (required by env cache)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["customer_id", "dt"]).reset_index(drop=True)

    # Reduce RAM (recommended)
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="raise").astype("int32")
    for c in ["load", "pv", "price"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float32")

    return df


def build_customer_segments(df: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """
    Build contiguous segments per customer_id (start,end) in global indices.
    Requires df sorted by customer_id, dt.
    """
    cust = df["customer_id"].to_numpy()
    change = np.flatnonzero(cust[1:] != cust[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(df)]
    cust_ids = cust[starts]
    # ensure contiguous
    if len(np.unique(cust_ids)) != len(cust_ids):
        raise ValueError("customer_id is not contiguous. Sort by ['customer_id','dt'] first.")
    return {int(cid): (int(s), int(e)) for cid, s, e in zip(cust_ids, starts, ends)}


def split_segment_train_test(seg_s: int, seg_e: int, train_frac: float = 0.8) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Simple chronological split within a customer segment:
      train: [seg_s, split)
      test:  [split, seg_e)
    """
    n = seg_e - seg_s
    split = seg_s + int(n * train_frac)
    # keep at least something in each side
    split = min(max(split, seg_s + 1), seg_e - 1)
    return (seg_s, split), (split, seg_e)


def build_start_pools_in_range(
    range_s: int,
    range_e: int,
    *,
    need_len: int,
    episode_len: int = 336,
    grid: int = 48,
) -> tuple[list[int], list[int]]:
    """
    Build candidate starts inside [range_s, range_e) such that [start, start+need_len) stays in range.
    Returns:
      - nonoverlap starts (stride episode_len)
      - random candidates (stride grid)
    """
    max_start = range_e - need_len
    if max_start < range_s:
        return [], []
    non = list(range(range_s, max_start + 1, episode_len))
    cand = list(range(range_s, max_start + 1, grid))
    return non, cand


# -----------------------------
# Balanced plan builder
# -----------------------------
def build_balanced_plan(
    cluster_to_customers: dict[int, list[int]],
    cust_train_pools: dict[int, dict[str, list[int]]],
    *,
    episodes_per_cluster: int,
    ratio_non: int = 2,
    ratio_rand: int = 1,
    per_customer_cap: int = 20,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """
    Returns schedule list of (customer_id, global_start_index) length = episodes_per_cluster * num_clusters
    Balanced by cluster and roughly balanced by customers (via per_customer_cap + round-robin).
    """
    rng = np.random.default_rng(seed)
    clusters = sorted(cluster_to_customers.keys())
    plan: list[tuple[int, int]] = []

    p_non = ratio_non / (ratio_non + ratio_rand)

    # Create per-customer shuffled "bags" so we can pop starts efficiently
    bags_non: dict[int, deque[int]] = {}
    bags_rand: dict[int, deque[int]] = {}

    def ensure_bags(cid: int):
        if cid not in bags_non:
            x = cust_train_pools[cid]["non"].copy()
            rng.shuffle(x)
            bags_non[cid] = deque(x)
        if cid not in bags_rand:
            y = cust_train_pools[cid]["rand"].copy()
            rng.shuffle(y)
            bags_rand[cid] = deque(y)

    for cl in clusters:
        custs = cluster_to_customers[cl]
        custs = [int(c) for c in custs]
        rng.shuffle(custs)

        used = defaultdict(int)
        cursor = 0
        quota = episodes_per_cluster
        guard = 0

        while quota > 0:
            guard += 1
            if guard > 200000:
                raise RuntimeError(f"Cannot fill quota for cluster {cl}. Not enough valid windows?")

            cid = custs[cursor % len(custs)]
            cursor += 1

            if cid not in cust_train_pools:
                continue
            if used[cid] >= per_customer_cap:
                continue

            ensure_bags(cid)

            want_non = (rng.random() < p_non)
            if want_non:
                if len(bags_non[cid]) == 0:
                    # refill
                    x = cust_train_pools[cid]["non"].copy()
                    rng.shuffle(x)
                    bags_non[cid] = deque(x)
                if len(bags_non[cid]) == 0:
                    continue
                s = bags_non[cid].popleft()
            else:
                if len(bags_rand[cid]) == 0:
                    y = cust_train_pools[cid]["rand"].copy()
                    rng.shuffle(y)
                    bags_rand[cid] = deque(y)
                if len(bags_rand[cid]) == 0:
                    continue
                s = bags_rand[cid].popleft()

            plan.append((cid, int(s)))
            used[cid] += 1
            quota -= 1

    rng.shuffle(plan)
    return plan


def eval_plan_from_test_pools(
    cluster_to_customers: dict[int, list[int]],
    cust_test_pools: dict[int, dict[str, list[int]]],
    *,
    episodes_per_cluster: int,
    episode_len: int,
    seed: int = 999,
) -> list[tuple[int, int]]:
    """
    Deterministic-ish evaluation plan: use non-overlap starts from test pools when possible.
    """
    rng = np.random.default_rng(seed)
    clusters = sorted(cluster_to_customers.keys())
    plan: list[tuple[int, int]] = []

    for cl in clusters:
        custs = [int(c) for c in cluster_to_customers[cl]]
        rng.shuffle(custs)
        quota = episodes_per_cluster
        cursor = 0
        guard = 0
        # import pdb;pdb.set_trace()

        while quota > 0:
            guard += 1
            if guard > 200000:
                raise RuntimeError(f"Cannot fill eval quota for cluster {cl}.")

            cid = custs[cursor % len(custs)]
            cursor += 1
            if cid not in cust_test_pools:
                continue

            non = cust_test_pools[cid]["non"]
            if len(non) == 0:
                continue

            # pick one start (rotate)
            s = non[quota % len(non)]
            plan.append((cid, int(s)))
            quota -= 1

    rng.shuffle(plan)
    return plan


def split_customer_train_1p5y_val_0p5y_test_1y(
    df: pd.DataFrame,
    seg_s: int,
    seg_e: int,
    *,
    dt_col: str = "dt",
):
    """
    Split per customer (within [seg_s, seg_e)) by time relative to first dt:
      Train: first 1.5 years
      Val:   next 0.5 years (i.e., up to 2.0 years)
      Test:  next 1.0 year (i.e., up to 3.0 years)

    Returns:
      train_range: (s,e) or None
      val_range:   (s,e) or None
      test_range:  (s,e) or None

    Notes:
      - If customer has less data than required windows, ranges may be None/shortened.
      - Ranges are index ranges in the original df (global indices), slicing style [start, end).
    """
    dti = pd.DatetimeIndex(df[dt_col].iloc[seg_s:seg_e])
    if len(dti) == 0:
        return None, None, None

    t0 = dti[0]

    t_train_end = t0 + pd.DateOffset(months=18)  # 1.5 years
    t_val_end   = t0 + pd.DateOffset(months=24)  # 2.0 years
    # t_test_end  = t0 + pd.DateOffset(months=36)  # 3.0 years

    # local indices inside segment
    i_train_end = int(dti.searchsorted(t_train_end, side="left"))
    i_val_end   = int(dti.searchsorted(t_val_end, side="left"))
    # i_test_end  = int(dti.searchsorted(t_test_end, side="left"))

    # convert to global indices
    g_train_end = seg_s + i_train_end
    g_val_end   = seg_s + i_val_end
    # g_test_end  = seg_s + i_test_end

    # clamp to [seg_s, seg_e]
    g_train_end = min(max(g_train_end, seg_s), seg_e)
    g_val_end   = min(max(g_val_end, seg_s), seg_e)
    # g_test_end  = min(max(g_test_end, seg_s), seg_e)

    # build ranges (must be non-empty to be meaningful)
    train_range = (seg_s, g_train_end) if g_train_end - seg_s > 1 else None
    val_range   = (g_train_end, g_val_end) if g_val_end - g_train_end > 1 else None
    # test_range  = (g_val_end, g_test_end) if g_test_end - g_val_end > 1 else None

    # If you want to force "test = rest of data after 2 years" instead of strictly 1 year:
    test_range = (g_val_end, seg_e) if seg_e - g_val_end > 1 else None

    return train_range, val_range, test_range

# =========================
# >>> ADDED: persistent balanced sampler (coverage for non-overlap across epochs)
# =========================
class PersistentBalancedSampler:
    """
    Build balanced (cluster) episode plans each epoch, but keep per-customer queues across epochs.
    - non-overlap starts: popped WITHOUT replacement across epochs until exhausted, then reshuffled.
      => improves coverage guarantee for train data.
    - random candidates: can also be queued (or treated similarly); no strict coverage needed.
    """
    def __init__(
        self,
        cluster_to_customers: dict[int, list[int]],
        pools: dict[int, dict[str, list[int]]],   # cust_train_pools
        *,
        ratio_non: int = 2,
        ratio_rand: int = 1,
        per_customer_cap: int = 20,
        seed: int = 0,
    ):
        self.cluster_to_customers = {int(k): [int(x) for x in v] for k, v in cluster_to_customers.items()}
        self.pools = pools
        self.rng = np.random.default_rng(seed)
        self.p_non = ratio_non / (ratio_non + ratio_rand)
        self.per_customer_cap = per_customer_cap

        # persistent cursors per cluster (round-robin continues across epochs)
        self.cursor = {cl: 0 for cl in self.cluster_to_customers.keys()}

        # persistent bags across epochs
        self.bags_non: dict[int, deque[int]] = {}
        self.bags_rand: dict[int, deque[int]] = {}

        # coverage tracker for non-overlap (small enough to track exactly)
        self.used_non: dict[int, set[int]] = defaultdict(set)

    def _ensure_bags(self, cid: int):
        if cid not in self.bags_non:
            x = self.pools[cid]["non"].copy()
            self.rng.shuffle(x)
            self.bags_non[cid] = deque(x)
        if cid not in self.bags_rand:
            y = self.pools[cid]["rand"].copy()
            self.rng.shuffle(y)
            self.bags_rand[cid] = deque(y)

    def _pop_non(self, cid: int) -> int | None:
        if cid not in self.pools:
            return None
        self._ensure_bags(cid)
        if len(self.bags_non[cid]) == 0:
            x = self.pools[cid]["non"].copy()
            self.rng.shuffle(x)
            self.bags_non[cid] = deque(x)
        if len(self.bags_non[cid]) == 0:
            return None
        s = int(self.bags_non[cid].popleft())
        self.used_non[cid].add(s)
        return s

    def _pop_rand(self, cid: int) -> int | None:
        if cid not in self.pools:
            return None
        self._ensure_bags(cid)
        if len(self.bags_rand[cid]) == 0:
            y = self.pools[cid]["rand"].copy()
            self.rng.shuffle(y)
            self.bags_rand[cid] = deque(y)
        if len(self.bags_rand[cid]) == 0:
            return None
        return int(self.bags_rand[cid].popleft())

    def sample_epoch(self, episodes_per_cluster: int) -> list[tuple[int, int]]:
        plan: list[tuple[int, int]] = []

        for cl, custs in self.cluster_to_customers.items():
            if len(custs) == 0:
                continue

            used_this_epoch = defaultdict(int)
            quota = episodes_per_cluster
            guard = 0
            # import pdb;pdb.set_trace()

            while quota > 0:
                guard += 1
                if guard > 300000:
                    raise RuntimeError(f"Cannot fill quota for cluster {cl}. Not enough valid windows?")

                cid = custs[self.cursor[cl] % len(custs)]
                self.cursor[cl] += 1

                if cid not in self.pools:
                    continue
                if used_this_epoch[cid] >= self.per_customer_cap:
                    continue

                want_non = (self.rng.random() < self.p_non)
                s = self._pop_non(cid) if want_non else self._pop_rand(cid)
                if s is None:
                    continue

                plan.append((cid, s))
                used_this_epoch[cid] += 1
                quota -= 1

        self.rng.shuffle(plan)
        return plan

    def non_overlap_coverage_report(self) -> dict[str, float]:
        """Coverage for non-overlap pools over whole training so far."""
        covs = []
        for cid, pools in self.pools.items():
            total = len(pools["non"])
            if total <= 0:
                continue
            covs.append(len(self.used_non[cid]) / total)
        if not covs:
            return {"mean": float("nan"), "min": float("nan"), "p10": float("nan"), "median": float("nan")}
        covs = np.array(covs, dtype=np.float32)
        return {
            "mean": float(np.mean(covs)),
            "min": float(np.min(covs)),
            "p10": float(np.percentile(covs, 10)),
            "median": float(np.percentile(covs, 50)),
        }

def split_plan(plan, n):
    return [plan[i::n] for i in range(n)]

def pad_subplans(subplans, rng):
    max_len = max(len(sp) for sp in subplans)
    flat = [x for sp in subplans for x in sp]
    for sp in subplans:
        while len(sp) < max_len:
            sp.append(flat[int(rng.integers(0, len(flat)))])
    return subplans

def save_checkpoint(path,name, policy, epoch, env_step, best_eval, sampler=None):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "env_step": env_step,
        "best_eval": best_eval,
        "policy": policy.state_dict(),
        "actor_optim": policy.actor_optim.state_dict(),
        "critic_optim": policy.critic_optim.state_dict(),
        # "sampler": sampler.state_dict() if sampler else None,
    }
    torch.save(ckpt, f"{path}/{name}")

def load_checkpoint(path, policy, sampler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.actor_optim.load_state_dict(ckpt["actor_optim"])
    policy.critic_optim.load_state_dict(ckpt["critic_optim"])
    return ckpt["epoch"], ckpt["env_step"], ckpt["best_eval"]


# -----------------------------
# Policy builder (shared by all training scripts)
# -----------------------------
def build_policy(env, alg: str, alg_param: dict | None = None, device: str = "cpu", epochs: int = 100):
    """Build a Tianshou policy (TD3 / SAC / DDPG) for the given env."""
    from tianshou.policy import DDPGPolicy, SACPolicy, TD3Policy
    from tianshou.exploration import GaussianNoise
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import Actor, ActorProb, Critic

    if alg_param is None:
        alg_param = {
            "tau": 0.005,
            "gamma": 0.99,
            "alpha": 0.2,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "update_actor_freq": 2,
        }

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    total_updates = epochs * 60 * int(env.episode_len_steps)

    net_a = Net(obs_dim, hidden_sizes=[256, 256], device=device)
    if alg == "SAC":
        actor = ActorProb(net_a, action_shape=act_dim, max_action=1.0, unbounded=False, device=device).to(device)
    else:
        actor = Actor(net_a, action_shape=act_dim, max_action=1.0, device=device).to(device)

    net_c1 = Net(obs_dim + act_dim, hidden_sizes=[256, 256], device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    net_c2 = Net(obs_dim + act_dim, hidden_sizes=[256, 256], device=device)
    critic2 = Critic(net_c2, device=device).to(device)

    actor_optim   = torch.optim.Adam(actor.parameters(),   lr=3e-4)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    actor_sched   = CosineAnnealingLR(actor_optim,   T_max=total_updates, eta_min=1e-5)
    critic1_sched = CosineAnnealingLR(critic1_optim, T_max=total_updates, eta_min=1e-5)
    critic2_sched = CosineAnnealingLR(critic2_optim, T_max=total_updates, eta_min=1e-5)

    if alg == "TD3":
        policy = TD3Policy(
            actor=actor, actor_optim=actor_optim,
            critic=critic1, critic_optim=critic1_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=alg_param["tau"], gamma=alg_param["gamma"],
            exploration_noise=GaussianNoise(sigma=0.1),
            policy_noise=alg_param["policy_noise"],
            noise_clip=alg_param["noise_clip"],
            update_actor_freq=alg_param["update_actor_freq"],
            action_space=env.action_space,
        ).to(device)
    elif alg == "SAC":
        target_entropy = float(-act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)
        policy = SACPolicy(
            actor=actor, actor_optim=actor_optim,
            critic=critic1, critic_optim=critic1_optim,
            critic2=critic2, critic2_optim=critic2_optim,
            tau=alg_param["tau"], gamma=alg_param["gamma"],
            alpha=(target_entropy, log_alpha, alpha_optim),
            action_space=env.action_space,
        ).to(device)
    elif alg == "DDPG":
        policy = DDPGPolicy(
            actor=actor, actor_optim=actor_optim,
            critic=critic1, critic_optim=critic1_optim,
            tau=alg_param["tau"], gamma=alg_param["gamma"],
            exploration_noise=GaussianNoise(sigma=0.1),
            action_space=env.action_space,
        ).to(device)
    else:
        raise ValueError(f"Unknown algorithm '{alg}'. Choose from: TD3, SAC, DDPG")

    return policy, [actor_sched, critic1_sched, critic2_sched]

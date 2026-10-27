"""One-time migration: register existing .pt checkpoints into the MLflow Model Registry.

Usage
-----
# Local mlruns/ folder (default)
python register_existing_checkpoints.py

# Remote MLflow server
python register_existing_checkpoints.py --mlflow_tracking_uri http://localhost:5000

# Different checkpoint directory
python register_existing_checkpoints.py --agent_dir /path/to/agent
"""

import argparse
import os
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import mlflow
import mlflow.tracking

# Maps agent prefix -> forecast style.
# Each prefix gets its OWN registry model name (hems-{prefix}-{alg}) to avoid
# collisions between N-* and V-* which both use point style but are distinct agents.
STYLE_MAP = {
    "I": "interval",  # interval forecast
    "N": "point",     # point forecast, trained with noise
    "V": "point",     # point forecast, vanilla (no noise in training)
    "R": "none",      # no forecast in observation
}

# Filename pattern: {PREFIX}-{ALG}-{SEED}.pt  e.g. I-TD3-3.pt
_PATTERN = re.compile(r"^([A-Z]+)-([A-Z]+[0-9]*)-(\d+)\.pt$")


def register_checkpoint(client, agent_dir: str, filename: str, dry_run: bool) -> str:
    """Register a single checkpoint. Returns a status string."""
    m = _PATTERN.match(filename)
    if not m:
        return f"SKIP  {filename}  (name doesn't match PREFIX-ALG-SEED.pt)"

    prefix, alg, seed = m.group(1), m.group(2), int(m.group(3))

    if prefix not in STYLE_MAP:
        return f"SKIP  {filename}  (unknown prefix '{prefix}' — add it to STYLE_MAP)"

    if alg not in ("TD3", "SAC", "DDPG"):
        return f"SKIP  {filename}  (unknown algorithm '{alg}')"

    style      = STYLE_MAP[prefix]
    model_name = f"hems-{prefix}-{alg}"   # prefix-based name avoids N/V collision
    agent_key  = f"{prefix}-{alg}"
    full_path  = os.path.join(agent_dir, filename)

    if dry_run:
        return f"DRY   {filename}  -> registry '{model_name}' seed={seed}"

    with mlflow.start_run(run_name=f"migration_{agent_key}_seed{seed}") as run:
        mlflow.set_tag("migration", "true")
        mlflow.set_tag("source_file", filename)
        mlflow.log_param("alg",           alg)
        mlflow.log_param("variant",       prefix)
        mlflow.log_param("forecast_style", style)
        mlflow.log_param("seed",          seed)
        mlflow.log_param("agent_key",     agent_key)
        mlflow.log_artifact(full_path, artifact_path="checkpoints")

        run_id = run.info.run_id
        mv = mlflow.register_model(f"runs:/{run_id}/checkpoints/{filename}", model_name)

    client.set_model_version_tag(model_name, mv.version, "seed",          str(seed))
    client.set_model_version_tag(model_name, mv.version, "agent_key",     agent_key)
    client.set_model_version_tag(model_name, mv.version, "variant",       prefix)
    client.set_model_version_tag(model_name, mv.version, "forecast_style", style)
    client.set_model_version_tag(model_name, mv.version, "migration",     "true")

    return f"OK    {filename}  -> '{model_name}' v{mv.version} seed={seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_dir", default="agent",
                    help="Directory containing the .pt checkpoint files (default: agent/)")
    ap.add_argument("--mlflow_tracking_uri", default=None,
                    help="MLflow tracking URI (default: local mlruns/)")
    ap.add_argument("--experiment", default="hems-migration",
                    help="MLflow experiment name to use for migration runs")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would be registered without actually doing it")
    args = ap.parse_args()

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = mlflow.tracking.MlflowClient()

    agent_dir = args.agent_dir
    if not os.path.isdir(agent_dir):
        print(f"ERROR: directory not found: {agent_dir}")
        return

    files = sorted(f for f in os.listdir(agent_dir) if f.endswith(".pt"))
    if not files:
        print(f"No .pt files found in {agent_dir}")
        return

    print(f"Found {len(files)} .pt files in '{agent_dir}'")
    if args.dry_run:
        print("DRY RUN — nothing will be written to MLflow\n")

    ok, skipped = 0, 0
    for filename in files:
        result = register_checkpoint(client, agent_dir, filename, args.dry_run)
        print(result)
        if result.startswith("OK"):
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} registered, {skipped} skipped.")
    if skipped:
        print("Add unknown prefixes to STYLE_MAP at the top of this file and re-run.")


if __name__ == "__main__":
    main()

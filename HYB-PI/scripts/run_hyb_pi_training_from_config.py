#!/usr/bin/env python3
"""
Run HYB-PI training from YAML configuration files.

Example with packaged input data:
  python scripts/run_hyb_pi_training_from_config.py \
    --paths-config config/paths/hyb_pi_paths_data_package.yaml \
    --training-config config/model/hyb_pi_training_config.yaml

Example with local machine-specific paths:
  python scripts/run_hyb_pi_training_from_config.py \
    --paths-config config/paths/hyb_pi_paths_local.yaml \
    --training-config config/model/hyb_pi_training_config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML file is empty: {path}")

    return cfg


def resolve_path(repo_root: Path, path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)


def main():
    parser = argparse.ArgumentParser(description="Run HYB-PI training from YAML configs.")
    parser.add_argument(
        "--paths-config",
        default="config/paths/hyb_pi_paths_data_package.yaml",
        help="Path configuration YAML.",
    )
    parser.add_argument(
        "--training-config",
        default="config/model/hyb_pi_training_config.yaml",
        help="Training configuration YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only; do not run training.",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    paths_cfg = load_yaml(repo_root / args.paths_config)
    train_cfg = load_yaml(repo_root / args.training_config)

    training = train_cfg["training"]
    outputs = train_cfg["outputs"]

    outdir = repo_root / outputs["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "src/training/train_hyb_pi.py"),
        "--csv",
        resolve_path(repo_root, paths_cfg["csv_file"]),
        "--outdir",
        str(outdir),
        "--f-par",
        str(training["f_par"]),
        "--n-trials",
        str(training["n_trials"]),
        "--seed",
        str(training["seed"]),
    ]

    if training.get("log_targets", False):
        cmd.append("--log-targets")

    if training.get("no_low_light_drop", False):
        cmd.append("--no-low-light-drop")

    print("\n[HYB-PI training command]\n")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] Command printed only. Training not started.")
        return

    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()

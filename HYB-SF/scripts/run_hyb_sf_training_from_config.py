#!/usr/bin/env python3
"""
Run HYB-SF training from YAML configuration files.

Example with packaged NetCDF input data:
  python scripts/run_hyb_sf_training_from_config.py \
    --paths-config config/paths/hyb_sf_paths_data_package.yaml \
    --training-config config/model/hyb_sf_training_config.yaml

Example with local machine-specific paths:
  python scripts/run_hyb_sf_training_from_config.py \
    --paths-config config/paths/hyb_sf_paths_local.yaml \
    --training-config config/model/hyb_sf_training_config.yaml
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
    parser = argparse.ArgumentParser(description="Run HYB-SF training from YAML configs.")
    parser.add_argument(
        "--paths-config",
        default="config/paths/hyb_sf_paths_data_package.yaml",
        help="Path configuration YAML.",
    )
    parser.add_argument(
        "--training-config",
        default="config/model/hyb_sf_training_config.yaml",
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

    templates = paths_cfg["templates"]
    time_period = train_cfg["time_period"]
    filters = train_cfg["filters"]
    weighting = train_cfg["weighting"]
    model_cfg = train_cfg["model"]
    outputs = train_cfg["outputs"]

    for key in [
        "model_out",
        "scaler_out",
        "bundle_out",
        "report_out",
        "diag_plot",
        "shap_beeswarm_plot",
    ]:
        out_path = repo_root / outputs[key]
        out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "src/training/train_hyb_sf.py"),

        "--ersem-root", resolve_path(repo_root, paths_cfg["ersem_root"]),
        "--scope-root", resolve_path(repo_root, paths_cfg["scope_root"]),
        "--par-root", resolve_path(repo_root, paths_cfg["par_root"]),
        "--sst-root", resolve_path(repo_root, paths_cfg["sst_root"]),
        "--wind-root", resolve_path(repo_root, paths_cfg["wind_root"]),
        "--sss-root", resolve_path(repo_root, paths_cfg["sss_root"]),
        "--depth-file", resolve_path(repo_root, paths_cfg["depth_file"]),

        "--ersem-tmpl", templates["ersem"],
        "--scope-tmpl", templates["scope"],
        "--par-tmpl", templates["par"],
        "--sst-tmpl", templates["sst"],
        "--wind-tmpl", templates["wind"],
        "--sss-tmpl", templates["sss"],

        "--start-year", str(time_period["start_year"]),
        "--start-month", str(time_period["start_month"]),
        "--end-year", str(time_period["end_year"]),
        "--end-month", str(time_period["end_month"]),
        "--train-end-year", str(time_period["train_end_year"]),
        "--val-end-year", str(time_period["val_end_year"]),

        "--min-ersem", str(filters["min_ersem"]),
        "--scale-min", str(filters["scale_min"]),
        "--scale-max", str(filters["scale_max"]),
        "--shelf-depth", str(filters["shelf_depth"]),

        "--tail-frac", str(weighting["tail_frac"]),
        "--tail-w", str(weighting["tail_w"]),
        "--random-seed", str(model_cfg["random_seed"]),

        "--model-out", str(repo_root / outputs["model_out"]),
        "--scaler-out", str(repo_root / outputs["scaler_out"]),
        "--bundle-out", str(repo_root / outputs["bundle_out"]),
        "--report-out", str(repo_root / outputs["report_out"]),
        "--diag-plot", str(repo_root / outputs["diag_plot"]),
        "--shap-beeswarm-plot", str(repo_root / outputs["shap_beeswarm_plot"]),
    ]

    if weighting.get("use_uncertainty", False):
        cmd.append("--use-uncertainty")
        if weighting.get("unc_max") is not None:
            cmd.extend(["--unc-max", str(weighting["unc_max"])])

    print("\n[HYB-SF training command]\n")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] Command printed only. Training not started.")
        return

    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()

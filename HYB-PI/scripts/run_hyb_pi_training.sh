#!/usr/bin/env bash
set -euo pipefail

# HYB-PI full training script.
#
# Expected input:
#   data/raw_external/Bouman_2017_merged_sstC_windSpeed_ssrd_full.csv

cd "$(dirname "$0")/.."

csv_file="data/raw_external/Bouman_2017_merged_sstC_windSpeed_ssrd_full.csv"

if [[ ! -f "${csv_file}" ]]; then
  echo "[HYB-PI] ERROR: Missing input CSV: ${csv_file}"
  exit 1
fi

echo "[HYB-PI] Input CSV found:"
ls -lh "${csv_file}"

echo "[HYB-PI] Starting full training..."

python scripts/run_hyb_pi_training_from_config.py \
  --paths-config config/paths/hyb_pi_paths_data_package.yaml \
  --training-config config/model/hyb_pi_training_config.yaml

echo "[HYB-PI] Training complete."
echo "[HYB-PI] Outputs:"
echo "  outputs/best_model_optuna.joblib"
echo "  outputs/optuna_study.joblib"
echo "  outputs/metrics.txt"
echo "  outputs/alpha_scatter.png"
echo "  outputs/pmb_scatter.png"
echo "  outputs/ersem_infer_stub.py"
echo "  outputs/shap/alpha_shap_beeswarm.png"
echo "  outputs/shap/pmb_shap_beeswarm.png"

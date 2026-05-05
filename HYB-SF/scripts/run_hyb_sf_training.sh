#!/usr/bin/env bash
set -euo pipefail

# HYB-SF full training script.
#
# Expected data layout:
#   data/raw_external/ERSEM/
#   data/raw_external/SCOPE/scope_on_ersem/
#   data/raw_external/FEATURES/PAR/unit_wm-2/
#   data/raw_external/FEATURES/SST/
#   data/raw_external/FEATURES/WINDSP10M/
#   data/raw_external/FEATURES/SSS/
#   data/raw_external/FEATURES/STATIC/depth_amm7.nc
#
# This reproduces the HYB-SF ML training setup:
#   2000-03 to 2015-12
#   train <= 2012
#   validation = 2013-2014
#   test = 2015
#   uncertainty weighting enabled
#   uncertainty cutoff = 200

cd "$(dirname "$0")/.."

echo "[HYB-SF] Checking required input data counts..."

n_ersem=$(find data/raw_external/ERSEM -type f -name "ersem_pp_*.nc" | wc -l)
n_scope=$(find data/raw_external/SCOPE/scope_on_ersem -type f -name "scope_gpp_on_ersem_*.nc" | wc -l)
n_par=$(find data/raw_external/FEATURES/PAR/unit_wm-2 -type f -name "par_on_ersem_24h_wm2_*.nc" | wc -l)
n_sst=$(find data/raw_external/FEATURES/SST -type f -name "sst_on_ersem_C_*.nc" | wc -l)
n_wind=$(find data/raw_external/FEATURES/WINDSP10M -type f -name "wind10m_on_ersem_ms_*.nc" | wc -l)
n_sss=$(find data/raw_external/FEATURES/SSS -type f -name "sss_on_ersem_psu_*.nc" | wc -l)

echo "  ERSEM: ${n_ersem}"
echo "  SCOPE: ${n_scope}"
echo "  PAR:   ${n_par}"
echo "  SST:   ${n_sst}"
echo "  WIND:  ${n_wind}"
echo "  SSS:   ${n_sss}"

if [[ "${n_ersem}" -ne 190 || "${n_scope}" -ne 190 || "${n_par}" -ne 190 || "${n_sst}" -ne 190 || "${n_wind}" -ne 190 || "${n_sss}" -ne 190 ]]; then
  echo "[HYB-SF] ERROR: Expected 190 monthly files for each HYB-SF input dataset."
  exit 1
fi

if [[ ! -f data/raw_external/FEATURES/STATIC/depth_amm7.nc ]]; then
  echo "[HYB-SF] ERROR: Missing static depth file: data/raw_external/FEATURES/STATIC/depth_amm7.nc"
  exit 1
fi

echo "[HYB-SF] Input data check passed."
echo "[HYB-SF] Starting full training..."

python scripts/run_hyb_sf_training_from_config.py \
  --paths-config config/paths/hyb_sf_paths_data_package.yaml \
  --training-config config/model/hyb_sf_training_config.yaml

echo "[HYB-SF] Training complete."
echo "[HYB-SF] Outputs:"
echo "  models/gpp_scale_model_log_shelf.joblib"
echo "  models/gpp_scale_scaler_log_shelf.joblib"
echo "  models/gpp_scale_bundle_log_shelf.joblib"
echo "  outputs/reports/gpp_scale_training_report_log_shelf.json"
echo "  outputs/diagnostics/gpp_scale_diagnostics.png"
echo "  outputs/shap/gpp_scale_shap_beeswarm.png"

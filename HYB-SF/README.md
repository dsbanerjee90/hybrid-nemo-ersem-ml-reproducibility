# HYB-SF: Hybrid Scale-Factor Model for Primary Production Correction

**Creator:** Deep S. Banerjee  
**Institution:** Plymouth Marine Laboratory, United Kingdom  
**Created:** November 2025  
**Project:** Hybrid machine-learning correction framework for coupled NEMO-FABM-ERSEM biogeochemical prediction  

This work was carried out at Plymouth Marine Laboratory as part of research on hybrid process-based and machine-learning approaches for marine ecosystem modelling.

**Copyright:** Copyright (c) 2025-2026 Deep S. Banerjee, Plymouth Marine Laboratory, and co-authors. All rights reserved unless otherwise stated.

This directory contains the HYB-SF machine-learning training workflow used in the hybrid NEMO-FABM-ERSEM modelling framework.

HYB-SF trains a machine-learning model to predict a log scale factor between satellite-derived gross primary production and baseline ERSEM gross primary production integrated to the euphotic zone:

    z = log(SCOPE_GPP / ERSEM_gpp_zeu)

The trained model is used to provide online scale-factor corrections to ERSEM primary production rates during coupled model integration.

## Directory structure

    HYB-SF/
    ├── config/
    │   ├── model/
    │   │   └── hyb_sf_training_config.yaml
    │   └── paths/
    │       ├── hyb_sf_paths_data_package.yaml
    │       └── hyb_sf_paths_template.yaml
    ├── data/
    │   ├── raw_external/
    │   └── metadata/
    ├── docs/
    │   └── MODEL_CARD_HYB_SF.md
    ├── scripts/
    │   ├── run_hyb_sf_training.sh
    │   └── run_hyb_sf_training_from_config.py
    └── src/
        └── training/
            └── train_hyb_sf.py

## Input data

The full training workflow expects NetCDF files under:

    data/raw_external/

Expected layout:

    data/raw_external/ERSEM/
    data/raw_external/SCOPE/scope_on_ersem/
    data/raw_external/FEATURES/PAR/unit_wm-2/
    data/raw_external/FEATURES/SST/
    data/raw_external/FEATURES/WINDSP10M/
    data/raw_external/FEATURES/SSS/
    data/raw_external/FEATURES/STATIC/depth_amm7.nc

For the full HYB-SF training period, 2000-03 to 2015-12, the package should contain:

    ERSEM monthly files: 190
    SCOPE monthly files: 190
    PAR monthly files:   190
    SST monthly files:   190
    WIND monthly files:  190
    SSS monthly files:   190
    Depth file:          1
    Total NetCDF files:  1141

The file manifest is provided in:

    data/metadata/hyb_sf_input_manifest.csv

## Python environment

Using conda:

    conda env create -f environment.yml
    conda activate hyb-sf

Or using pip:

    pip install -r requirements.txt

## Run full HYB-SF training

    bash scripts/run_hyb_sf_training.sh

This script checks the required input file counts and then runs:

    python scripts/run_hyb_sf_training_from_config.py \
      --paths-config config/paths/hyb_sf_paths_data_package.yaml \
      --training-config config/model/hyb_sf_training_config.yaml

The training configuration reproduces the HYB-SF setup:

    Training period:      2000-03 to 2012-12
    Validation period:    2013-01 to 2014-12
    Test period:          2015-01 to 2015-12
    Shelf mask:           depth <= 200 m
    Target:               log(SCOPE_GPP / ERSEM_gpp_zeu)
    Uncertainty enabled:  yes
    Uncertainty cutoff:   200
    Model:                HistGradientBoostingRegressor

## Outputs

The training script creates:

    models/gpp_scale_model_log_shelf.joblib
    models/gpp_scale_scaler_log_shelf.joblib
    models/gpp_scale_bundle_log_shelf.joblib

    outputs/reports/gpp_scale_training_report_log_shelf.json
    outputs/diagnostics/gpp_scale_diagnostics.png
    outputs/shap/gpp_scale_shap_beeswarm.png

The diagnostics include the true-vs-predicted scale-factor scatter plot and the SHAP beeswarm plot.

# HYB-PI: Hybrid Photosynthesis-Irradiance Parameter Model

Creator: Deep S. Banerjee  
Institution: Plymouth Marine Laboratory, United Kingdom  
Created: November 2025  
Project: Hybrid machine-learning correction framework for coupled NEMO-FABM-ERSEM biogeochemical prediction

This work was carried out at Plymouth Marine Laboratory as part of research on hybrid process-based and machine-learning approaches for marine ecosystem modelling.

Copyright (c) 2025-2026 Deep S. Banerjee, Plymouth Marine Laboratory, and co-authors. All rights reserved unless otherwise stated.

## Overview

HYB-PI trains machine-learning emulators for phytoplankton photosynthesis-irradiance parameters using the Bouman et al. experimental dataset merged with environmental predictors.

The workflow trains two Optuna-tuned HistGradientBoostingRegressor models:

    alpha model: predicts alpha_raw
    PBmax model: predicts pbmax_raw

The trained models are designed for use in the hybrid NEMO-FABM-ERSEM framework, where predicted PI parameters can dynamically inform primary production calculations.

## Input data

The packaged input CSV should be located at:

    data/raw_external/Bouman_2017_merged_sstC_windSpeed_ssrd_full.csv

The CSV must contain the following required columns:

    Latitude
    Longitude
    Depth water [m]
    Date/Time
    alpha [(mg C/mg Chl a/h)/(µE/m**2/s)]
    PBmax [mg C/mg Chl a/h]
    ssrd_J_m2
    sst_C
    wind_speed_ms

## Feature set

The model uses the following 10+1 feature setup:

    sin_lon
    cos_lon
    Latitude
    depth_m
    daylength_h
    sin_doy
    cos_doy
    sst_C
    wind_speed_ms
    PAR_Wm2
    sst_x_PAR

PAR is calculated from daily SSRD using:

    PAR_Wm2 = (ssrd_J_m2 / 86400) * 0.45

## Data gates and filtering

The training workflow applies:

    depth_m <= 200 m
    alpha_raw <= 0.12
    daylength_h >= 2 h
    PAR_Wm2 >= 5 W m-2

The low-light filter can be disabled in the configuration, but the default packaged configuration keeps it enabled.

## Training setup

The default configuration reproduces the HYB-PI setup:

    log_targets: true
    f_par: 0.45
    n_trials: 50
    seed: 42
    model: dual HistGradientBoostingRegressor
    hyperparameter tuning: Optuna TPESampler(seed=42)

The workflow uses an 80/20 train-holdout split and reports 5-fold cross-validation using the best hyperparameters.

## Python environment

Using conda:

    conda env create -f environment.yml
    conda activate hyb-pi

Or using pip:

    pip install -r requirements.txt

## Run full HYB-PI training

    bash scripts/run_hyb_pi_training.sh

This runs:

    python scripts/run_hyb_pi_training_from_config.py \
      --paths-config config/paths/hyb_pi_paths_data_package.yaml \
      --training-config config/model/hyb_pi_training_config.yaml

## Outputs

The workflow creates:

    outputs/best_model_optuna.joblib
    outputs/optuna_study.joblib
    outputs/metrics.txt
    outputs/alpha_scatter.png
    outputs/pmb_scatter.png
    outputs/ersem_infer_stub.py
    outputs/shap/alpha_shap_beeswarm.png
    outputs/shap/pmb_shap_beeswarm.png

Large input data and generated model/output artefacts are excluded from Git tracking and should be distributed through the associated reproducibility package rather than committed to the code repository.

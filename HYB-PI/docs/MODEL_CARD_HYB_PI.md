# Model Card: HYB-PI

## Model name

HYB-PI: Hybrid Photosynthesis-Irradiance Parameter Model

## Creator

Deep S. Banerjee

## Institution

Plymouth Marine Laboratory, United Kingdom

## Created

November 2025

## Purpose

HYB-PI predicts phytoplankton photosynthesis-irradiance parameters alpha and PBmax from environmental and spatio-temporal predictors. The model is intended for integration into the hybrid NEMO-FABM-ERSEM framework.

## Model type

Two Optuna-tuned HistGradientBoostingRegressor models from scikit-learn:

    alpha_model
    pmb_model

## Targets

    alpha_raw
    pbmax_raw

When log-target training is enabled, the models are trained on:

    log10(max(alpha_raw, eps))
    log10(max(pbmax_raw, eps))

and predictions are converted back to physical space.

## Input features

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

## Feature engineering

Longitude is represented using sine and cosine harmonics.

Day of year is represented using sine and cosine seasonal harmonics.

Daylength is calculated from latitude and day of year.

PAR is derived from daily SSRD:

    PAR_Wm2 = (ssrd_J_m2 / 86400) * 0.45

The interaction feature is:

    sst_x_PAR = sst_C * PAR_Wm2

## Training domain

The default training domain is restricted to:

    depth_m <= 200 m
    alpha_raw <= 0.12

The default configuration also drops very low-light samples:

    daylength_h >= 2 h
    PAR_Wm2 >= 5 W m-2

## Training setup

    log_targets = true
    f_par = 0.45
    n_trials = 50
    seed = 42
    Optuna sampler = TPESampler(seed=42)

## Outputs

    outputs/best_model_optuna.joblib
    outputs/optuna_study.joblib
    outputs/metrics.txt
    outputs/alpha_scatter.png
    outputs/pmb_scatter.png
    outputs/ersem_infer_stub.py
    outputs/shap/alpha_shap_beeswarm.png
    outputs/shap/pmb_shap_beeswarm.png

## Intended use

HYB-PI is intended for hybrid marine ecosystem modelling experiments where machine-learning-predicted PI parameters are used to dynamically inform primary production calculations inside NEMO-FABM-ERSEM.

## Interpretability

SHAP beeswarm plots are generated separately for the alpha and PBmax models:

    outputs/shap/alpha_shap_beeswarm.png
    outputs/shap/pmb_shap_beeswarm.png

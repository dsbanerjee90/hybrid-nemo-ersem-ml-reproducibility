# Model Card: HYB-SF

## Model name

HYB-SF: Hybrid Scale-Factor Model

## Purpose

HYB-SF predicts a multiplicative scale factor for ERSEM gross primary production over shelf regions. The model is trained to estimate the log ratio between satellite-derived gross primary production and baseline ERSEM gross primary production integrated to the euphotic zone.

    z = log(SCOPE_GPP / ERSEM_gpp_zeu)

During coupled model integration, the predicted scale factor is used to adjust primary production process rates online while preserving the native ERSEM stoichiometric framework.

## Model type

HistGradientBoostingRegressor from scikit-learn.

## Target

    z = log(SCOPE_GPP / ERSEM_gpp_zeu)

where:

    SCOPE_GPP      = satellite-derived gross primary production on the ERSEM grid
    ERSEM_gpp_zeu  = baseline ERSEM gross primary production integrated to the euphotic zone

## Input features

The model uses monthly shelf-sea predictors available or reconstructable during runtime:

    sst
    par
    wind10m
    sss
    lon
    lat
    depth
    daylength_h
    par_day
    wind2
    log1p_depth
    sst_x_par
    month_sin
    month_cos

## Training period and split

    Training:    2000-03 to 2012-12
    Validation:  2013-01 to 2014-12
    Test:        2015-01 to 2015-12

## Spatial domain

Training is restricted to shelf regions:

    0 < depth <= 200 m

## Data filtering

Samples are retained only where all required predictors are finite and:

    ERSEM_gpp_zeu >= 1e-3
    SCOPE_GPP > 0
    0.05 <= SCOPE_GPP / ERSEM_gpp_zeu <= 6.0

## Uncertainty handling

SCOPE uncertainty is not used as an input feature.

When enabled, uncertainty is used in two ways:

1. Optional filtering using unc_max = 200.
2. Training weights that down-weight samples with higher uncertainty.

The training configuration uses:

    use_uncertainty = true
    unc_max = 200

## Tail weighting

The upper tail of the linear scale-factor distribution is up-weighted during training:

    tail_frac = 0.25
    tail_weight = 3.0

## Model hyperparameters

    loss = squared_error
    learning_rate = 0.06
    max_depth = 6
    max_iter = 400
    l2_regularization = 1e-3
    min_samples_leaf = 100
    validation_fraction = 0.1
    early_stopping = true
    random_state = 42

## Outputs

The workflow produces:

    models/gpp_scale_model_log_shelf.joblib
    models/gpp_scale_scaler_log_shelf.joblib
    models/gpp_scale_bundle_log_shelf.joblib

    outputs/reports/gpp_scale_training_report_log_shelf.json
    outputs/diagnostics/gpp_scale_diagnostics.png
    outputs/shap/gpp_scale_shap_beeswarm.png

## Intended use

The model is intended for shelf-sea hybrid biogeochemical modelling experiments where data-informed corrections are applied to primary production process rates in NEMO-FABM-ERSEM.

## Interpretability

SHAP analysis is used to assess feature contributions. The generated SHAP beeswarm plot is saved to:

    outputs/shap/gpp_scale_shap_beeswarm.png

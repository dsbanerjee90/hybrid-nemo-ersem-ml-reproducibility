# Hybrid NEMO-FABM-ERSEM ML Reproducibility Package

Creator: Deep S. Banerjee  
Institution: Plymouth Marine Laboratory, United Kingdom  
Created: November 2025  

This repository contains the reproducible machine-learning workflows associated with the hybrid NEMO-FABM-ERSEM modelling framework described in Banerjee et al. (2026), "Online machine-learning corrections improve coupled ocean biogeochemical prediction".

The repository contains two model workflows:

## 1. HYB-SF

HYB-SF is the hybrid scale-factor model. It predicts a log scale factor between satellite-derived gross primary production and baseline ERSEM gross primary production integrated to the euphotic zone.

Directory:

    HYB-SF/

Main run command:

    cd HYB-SF
    bash scripts/run_hyb_sf_training.sh

## 2. HYB-PI

HYB-PI is the hybrid photosynthesis-irradiance parameter model. It trains machine-learning emulators for alpha and PBmax using a Bouman et al. PI-parameter dataset merged with environmental predictors.

Directory:

    HYB-PI/

Main run command:

    cd HYB-PI
    bash scripts/run_hyb_pi_training.sh

## Data and generated outputs

Large input data, trained model artefacts and generated outputs are not committed to GitHub. They are provided separately as part of the reproducibility/data package.

Expected data locations after unpacking the data package:

    HYB-SF/data/raw_external/
    HYB-PI/data/raw_external/

Generated outputs are written to:

    HYB-SF/models/
    HYB-SF/outputs/
    HYB-PI/outputs/

## Notes

The code is provided for scholarly review and reproducibility of the associated manuscript. Original source-data ownership, licences and citation requirements remain with the original data providers.

# Hybrid NEMO-FABM-ERSEM ML Reproducibility Package

Creator: Deep S. Banerjee  
Institution: Plymouth Marine Laboratory, United Kingdom  

This repository contains the reproducible machine-learning workflows associated with the hybrid NEMO-FABM-ERSEM modelling framework described in Banerjee et al. (2026), **“Online machine-learning corrections improve coupled ocean biogeochemical prediction.”**

The repository contains two complementary hybrid model workflows: **HYB-SF** and **HYB-PI**.

## HYB-SF

HYB-SF is the hybrid scale-factor model. It predicts a log scale factor between satellite-derived gross primary production and baseline ERSEM gross primary production integrated to the euphotic zone.

Directory:

    HYB-SF/

Main training command:

    cd HYB-SF
    bash scripts/run_hyb_sf_training.sh

The v1.1.0 repository includes the training code, configuration files, trained model artefacts, training report, diagnostic plots, SHAP analysis, input-data manifest and helper utilities.

The full HYB-SF training-input dataset is not stored directly in GitHub because of its size. It is distributed separately with the associated Zenodo reproducibility release as:

    HYB-SF-training-inputs-v1.1.0.tar.gz

The SHA-256 checksum is provided in:

    HYB-SF-training-inputs-v1.1.0.sha256

SHA-256:

    febbcdf3c6eba4d15ba5191599b4505f966039feb5305b6438ca0552dc0ea6a2

After downloading and extracting the archive, place the input data under:

    HYB-SF/data/raw_external/

## HYB-PI

HYB-PI is the hybrid photosynthesis-irradiance parameter model. It trains machine-learning emulators for the photosynthetic parameters alpha and PBmax using the Bouman et al. dataset merged with environmental predictors.

Directory:

    HYB-PI/

Main training command:

    cd HYB-PI
    bash scripts/run_hyb_pi_training.sh

The v1.1.0 repository includes the training code, configuration files, trained model artefacts, Optuna study, performance metrics, diagnostic plots and SHAP analyses.

The HYB-PI input dataset used by the reproducibility workflow is included under:

    HYB-PI/data/raw_external/

## Repository and archival release

Browsable source-code repository:

    https://github.com/dsbanerjee90/hybrid-nemo-ersem-ml-reproducibility

The original v1.0.0 archival release is available on Zenodo:

    https://zenodo.org/records/23083708

For v1.1.0, the GitHub release is paired with an updated Zenodo version containing the complete archival reproducibility package and the separately packaged HYB-SF training inputs.

## Source data for the publication

Numerical source data underlying the main and Supplementary Figures and Tables are supplied separately with the journal article as **Supplementary Data 1**. They are intentionally kept separate from this model-training reproducibility repository.

## Software environment

Environment specifications and Python requirements are provided separately for HYB-SF and HYB-PI:

    HYB-SF/environment.yml
    HYB-SF/requirements.txt
    HYB-PI/environment.yml
    HYB-PI/requirements.txt

## Licence and data provenance

The code in this repository is released under the BSD 3-Clause License. See the `LICENSE` file for details.

Original source-data ownership, licences and citation requirements remain with the original data providers. Inclusion of third-party-derived data in the reproducibility package does not alter the terms of the original data products.

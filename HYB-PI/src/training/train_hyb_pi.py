#!/usr/bin/env python3
"""
HYB-PI training script
======================

Creator
-------
Deep S. Banerjee

Institution
-----------
Plymouth Marine Laboratory, United Kingdom

Created
-------
November 2025

Copyright
---------
Copyright (c) 2025-2026 Deep S. Banerjee, Plymouth Marine Laboratory,
and co-authors. All rights reserved unless otherwise stated.

Project context
---------------
This work was carried out at Plymouth Marine Laboratory as part of the
development of a hybrid process-based and machine-learning framework for
coupled NEMO-FABM-ERSEM marine biogeochemical prediction.

Scientific purpose
------------------
Optuna-based dual HGBR trainer (alpha & PBmax) with 10+1 feature setup.

Features used:
  ['sin_lon', 'cos_lon', 'Latitude', 'depth_m',
   'daylength_h', 'sin_doy', 'cos_doy',
   'sst_C', 'wind_speed_ms', 'PAR_Wm2', 'sst_x_PAR']

Gates:
  - depth_m <= 200 m
  - alpha_raw <= 0.12

Optimisation:
  - Hyperparameters of HistGradientBoostingRegressor
  - Joint objective: maximise mean R2 of alpha & PBmax on validation

Outputs:
  - best_model_optuna.joblib   (alpha_model, pmb_model, preprocessor, feat_cols, etc.)
  - optuna_study.joblib
  - metrics.txt
  - alpha_scatter.png, pmb_scatter.png
  - alpha_shap_beeswarm.png, pmb_shap_beeswarm.png
  - ersem_infer_stub.py        (for NEMO-ERSEM integration)
"""

import argparse, os, math, warnings
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error

import optuna

# Optional SHAP (not used inside Optuna loop, but can still be used later manually)
try:
    import shap
except Exception:
    shap = None

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- helpers ----------
def daylength_hours(lat_deg: float, doy: int) -> float:
    lat = math.radians(float(lat_deg))
    g = 2.0*math.pi*(int(doy)-1)/365.0
    dec = (0.006918 - 0.399912*math.cos(g) + 0.070257*math.sin(g)
           - 0.006758*math.cos(2*g) + 0.000907*math.sin(2*g)
           - 0.002697*math.cos(3*g) + 0.00148*math.sin(3*g))
    cwo = -math.tan(lat)*math.tan(dec)
    if cwo >= 1.0:  return 0.0
    if cwo <= -1.0: return 24.0
    return 24.0*math.acos(cwo)/math.pi

def fourier_lon(lon):
    lonr = np.deg2rad(lon)
    return np.sin(lonr), np.cos(lonr)

def season_harmonics(doy):
    ang = 2.0 * math.pi * (float(int(doy)) / 365.0)
    return math.sin(ang), math.cos(ang)

def load_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'Depth water [m]': 'depth_m',
        'Date/Time': 'datetime',
        'alpha [(mg C/mg Chl a/h)/(µE/m**2/s)]': 'alpha_raw',
        'PBmax [mg C/mg Chl a/h]': 'pbmax_raw',
        'sst_C': 'sst_C',
        'wind_speed_ms': 'wind_speed_ms',
        'ssrd_J_m2': 'ssrd_J_m2',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    })
    for c in ['Latitude', 'Longitude', 'depth_m', 'alpha_raw', 'pbmax_raw',
              'sst_C', 'wind_speed_ms', 'ssrd_J_m2']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude', 'depth_m',
                           'alpha_raw', 'pbmax_raw', 'datetime'])
    return df

def build_features(df, f_par=0.45, drop_low_light=True):
    """
    Physics-guided feature engineering.

    - depth_m <= 200 m
    - Features:
        sin_lon, cos_lon, Latitude, depth_m,
        daylength_h, sin_doy, cos_doy,
        sst_C, wind_speed_ms, PAR_Wm2, sst_x_PAR
    """
    df = df.copy()
    df = df[df['depth_m'] <= 200.0]

    # hygiene on depth to mirror training domain tightly
    df['depth_m'] = df['depth_m'].clip(lower=0.0, upper=200.0)  # <<< ADDED

    df['doy'] = df['datetime'].dt.dayofyear.astype(int)
    df['daylength_h'] = [daylength_hours(lt, d)
                         for lt, d in zip(df['Latitude'], df['doy'])]

    sL, cL = fourier_lon(df['Longitude'].to_numpy())
    df['sin_lon'] = sL
    df['cos_lon'] = cL
    df['sin_doy'], df['cos_doy'] = zip(*[season_harmonics(d)
                                         for d in df['doy']])

    # SSRD[J/m2/day] → SW[W/m2]=/86400 → PAR ≈ f_par × SW
    df['PAR_Wm2'] = (df['ssrd_J_m2'] / 86400.0) * float(f_par)

    # Hygiene/clipping
    df['sst_C'] = df['sst_C'].clip(-2, 35)
    df['wind_speed_ms'] = df['wind_speed_ms'].clip(0, 25)
    df['PAR_Wm2'] = df['PAR_Wm2'].clip(0, 1400)

    # Interaction: SST × PAR (thermal-light environment)
    df['sst_x_PAR'] = df['sst_C'] * df['PAR_Wm2']

    if drop_low_light:
        df = df[(df['daylength_h'] >= 2) & (df['PAR_Wm2'] >= 5)].copy()

    feat_cols = [
        'sin_lon', 'cos_lon', 'Latitude',
        'depth_m',                 # <<< ADDED
        'daylength_h', 'sin_doy', 'cos_doy',
        'sst_C', 'wind_speed_ms', 'PAR_Wm2', 'sst_x_PAR'
    ]

    df = df.dropna(subset=feat_cols + ['alpha_raw', 'pbmax_raw'])
    return df, feat_cols

def scatter_1to1(y_true, y_pred, title, outpng):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    plt.plot([lo, hi], [lo, hi], lw=1)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()

def residuals_plot(y_true, y_pred, title, outpng):
    res = y_pred - y_true
    plt.figure(figsize=(6, 3.5))
    plt.scatter(y_pred, res, s=10, alpha=0.6)
    plt.axhline(0.0, lw=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()


def shap_beeswarm_plot(model, X_scaled, feat_cols, title, outpng, max_samples=2000, seed=42):
    """
    SHAP beeswarm plot for a fitted HistGradientBoostingRegressor.

    Parameters
    ----------
    model : fitted sklearn estimator
        Fitted HGBR model.
    X_scaled : np.ndarray
        Scaled feature matrix used for model training/evaluation.
    feat_cols : list[str]
        Feature names in the same order as X_scaled.
    title : str
        Plot title.
    outpng : str
        Output PNG path.
    max_samples : int
        Maximum number of samples used for SHAP calculation.
    seed : int
        Random seed for subsampling.
    """
    if shap is None:
        print("[shap] shap not installed; skipping:", outpng)
        return

    n = X_scaled.shape[0]
    if n == 0:
        print("[shap] no samples available; skipping:", outpng)
        return

    rng = np.random.default_rng(seed)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_used = X_scaled[idx]
    else:
        X_used = X_scaled

    print(f"[shap] Computing SHAP beeswarm for {title} using N={X_used.shape[0]} samples")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_used)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_used,
        feature_names=feat_cols,
        plot_type="dot",
        show=False,
        max_display=len(feat_cols),
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close()

    print("[shap] Saved:", outpng)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--f-par", type=float, default=0.45)
    ap.add_argument("--log-targets", action="store_true")
    ap.add_argument("--no-low-light-drop", action="store_true",
                    help="keep points with very low daylength/PAR")
    ap.add_argument("--n-trials", type=int, default=50,
                    help="number of Optuna trials")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    eps = 1e-5

    # Build dataset
    df = load_csv(args.csv)
    df, feat_cols = build_features(
        df, f_par=args.f_par,
        drop_low_light=(not args.no_low_light_drop)
    )

    # Train BOTH alpha and PBmax ONLY on rows where alpha_raw <= 0.12
    df = df[df["alpha_raw"] <= 0.12].copy()

    X = df[feat_cols].copy()
    y_alpha_phys = df['alpha_raw'].astype(float).to_numpy()
    y_pmb_phys   = df['pbmax_raw'].astype(float).to_numpy()

    log_targets = bool(args.log_targets)
    if log_targets:
        y_alpha = np.log10(np.maximum(y_alpha_phys, eps))
        y_pmb   = np.log10(np.maximum(y_pmb_phys,   eps))
    else:
        y_alpha = y_alpha_phys.copy()
        y_pmb   = y_pmb_phys.copy()

    # Fixed train/validation split (used consistently)
    Xtr_df, Xval_df, ya_tr, ya_val, yp_tr, yp_val = train_test_split(
        X, y_alpha, y_pmb, test_size=0.2, random_state=args.seed
    )

    pre = ColumnTransformer([("num", StandardScaler(), feat_cols)],
                            sparse_threshold=0.0)
    pre.fit(Xtr_df)
    Xtr = pre.transform(Xtr_df)
    Xval = pre.transform(Xval_df)

    # --------- Optuna objective ---------
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space for HGBR
        loss = trial.suggest_categorical("loss", ["squared_error", "absolute_error"])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_iter = trial.suggest_int("max_iter", 200, 900)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 255)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 200)
        l2_reg = trial.suggest_float("l2_regularization", 1e-8, 1e-1, log=True)
        # Depth: include None as an option
        max_depth = trial.suggest_categorical("max_depth", [None, 3, 5, 7, 9])

        common_kwargs = dict(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg,
            max_depth=max_depth,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30
        )

        m_alpha = HistGradientBoostingRegressor(
            random_state=41,
            **common_kwargs
        )
        m_pmb = HistGradientBoostingRegressor(
            random_state=42,
            **common_kwargs
        )

        m_alpha.fit(Xtr, ya_tr)
        m_pmb.fit(Xtr, yp_tr)

        # Validation predictions (back to physical space if needed)
        a_hat = m_alpha.predict(Xval)
        p_hat = m_pmb.predict(Xval)
        if log_targets:
            a_hat_phys = 10.0**a_hat - eps
            a_true_phys = 10.0**ya_val - eps
            p_hat_phys = 10.0**p_hat - eps
            p_true_phys = 10.0**yp_val - eps
        else:
            a_hat_phys = a_hat; a_true_phys = ya_val
            p_hat_phys = p_hat; p_true_phys = yp_val

        a_r2 = r2_score(a_true_phys, a_hat_phys)
        p_r2 = r2_score(p_true_phys, p_hat_phys)

        # Objective: maximise mean R2, so we minimise negative mean R2
        mean_r2 = 0.5 * (a_r2 + p_r2)
        trial.set_user_attr("alpha_R2", float(a_r2))
        trial.set_user_attr("pmb_R2", float(p_r2))
        return -mean_r2

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    # Save study
    study_path = os.path.join(args.outdir, "optuna_study.joblib")
    joblib.dump(study, study_path, compress=3)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_alpha_R2 = best_trial.user_attrs.get("alpha_R2", None)
    best_pmb_R2 = best_trial.user_attrs.get("pmb_R2", None)

    print("Best trial:", best_trial.number)
    print("  Params:", best_params)
    print("  alpha_R2 (val):", best_alpha_R2)
    print("  pmb_R2   (val):", best_pmb_R2)

    # ---------- Retrain best models on TRAIN ONLY (honest holdout) ----------
    pre_best = pre        # scaler fitted only on training data
    Xtr_best = Xtr

    common_kwargs_best = dict(
        loss=best_params["loss"],
        learning_rate=best_params["learning_rate"],
        max_iter=best_params["max_iter"],
        max_leaf_nodes=best_params["max_leaf_nodes"],
        min_samples_leaf=best_params["min_samples_leaf"],
        l2_regularization=best_params["l2_regularization"],
        max_depth=best_params["max_depth"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30
    )

    m_alpha_best = HistGradientBoostingRegressor(
        random_state=41,
        **common_kwargs_best
    )
    m_pmb_best = HistGradientBoostingRegressor(
        random_state=42,
        **common_kwargs_best
    )

    m_alpha_best.fit(Xtr_best, ya_tr)
    m_pmb_best.fit(Xtr_best, yp_tr)

    # For honest holdout metrics/plots use Xval (scaled with pre_best)
    Xval_scaled = pre_best.transform(Xval_df)
    a_hat_val = m_alpha_best.predict(Xval_scaled)
    p_hat_val = m_pmb_best.predict(Xval_scaled)
    if log_targets:
        a_hat_val_phys = 10.0**a_hat_val - eps
        a_true_val_phys = 10.0**ya_val - eps
        p_hat_val_phys = 10.0**p_hat_val - eps
        p_true_val_phys = 10.0**yp_val - eps
    else:
        a_hat_val_phys = a_hat_val; a_true_val_phys = ya_val
        p_hat_val_phys = p_hat_val; p_true_val_phys = yp_val

    a_r2_holdout = r2_score(a_true_val_phys, a_hat_val_phys)
    p_r2_holdout = r2_score(p_true_val_phys, p_hat_val_phys)
    a_mae_holdout = mean_absolute_error(a_true_val_phys, a_hat_val_phys)
    p_mae_holdout = mean_absolute_error(p_true_val_phys, p_hat_val_phys)

    # ---------- 5-fold CV with best hyperparameters (for reporting only) ----------
    X_all = X.reset_index(drop=True)
    y_alpha_all = y_alpha.copy()
    y_pmb_all = y_pmb.copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    r2a, r2p, maa, mapb = [], [], [], []
    for tr, va in kf.split(X_all):
        pre_k = ColumnTransformer([("num", StandardScaler(), feat_cols)],
                                  sparse_threshold=0.0)
        pre_k.fit(X_all.iloc[tr, :])
        Xtr_k = pre_k.transform(X_all.iloc[tr, :])
        Xva_k = pre_k.transform(X_all.iloc[va, :])

        ma_k = HistGradientBoostingRegressor(
            random_state=41,
            **common_kwargs_best
        )
        mp_k = HistGradientBoostingRegressor(
            random_state=42,
            **common_kwargs_best
        )
        ma_k.fit(Xtr_k, y_alpha_all[tr])
        mp_k.fit(Xtr_k, y_pmb_all[tr])

        a_k = ma_k.predict(Xva_k)
        p_k = mp_k.predict(Xva_k)
        if log_targets:
            a_k_phys = 10.0**a_k - eps; p_k_phys = 10.0**p_k - eps
            a_t = 10.0**y_alpha_all[va] - eps; p_t = 10.0**y_pmb_all[va] - eps
        else:
            a_k_phys = a_k; p_k_phys = p_k
            a_t = y_alpha_all[va]; p_t = y_pmb_all[va]

        r2a.append(r2_score(a_t, a_k_phys))
        r2p.append(r2_score(p_t, p_k_phys))
        maa.append(mean_absolute_error(a_t, a_k_phys))
        mapb.append(mean_absolute_error(p_t, p_k_phys))

    outdir = args.outdir
    shap_dir = os.path.join(outdir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    # Save pack (train-only model + scaler)
    model_path = os.path.join(outdir, "best_model_optuna.joblib")
    pack = {
        "alpha_model":   m_alpha_best,
        "pmb_model":     m_pmb_best,
        "preprocessor":  pre_best,
        "feat_cols":     feat_cols,
        "log_targets":   log_targets,
        "eps":           float(eps),
        "f_par":         float(args.f_par),
        "alpha_cap":     0.12,
        "optuna_best_params": best_params
    }
    joblib.dump(pack, model_path, compress=3)

    # Plots using honest holdout
    scatter_1to1(a_true_val_phys, a_hat_val_phys,
                 f"alpha holdout (α≤0.12, depth≤200m) R²={a_r2_holdout:.3f}",
                 os.path.join(outdir, "alpha_scatter.png"))
    scatter_1to1(p_true_val_phys, p_hat_val_phys,
                 f"Pmb holdout (α≤0.12, depth≤200m) R²={p_r2_holdout:.3f}",
                 os.path.join(outdir, "pmb_scatter.png"))

    # SHAP beeswarm plots using the train-only fitted models
    shap_beeswarm_plot(
        m_alpha_best,
        Xtr_best,
        feat_cols,
        "HYB-PI alpha SHAP beeswarm",
        os.path.join(shap_dir, "alpha_shap_beeswarm.png"),
        seed=args.seed,
    )

    shap_beeswarm_plot(
        m_pmb_best,
        Xtr_best,
        feat_cols,
        "HYB-PI PBmax SHAP beeswarm",
        os.path.join(shap_dir, "pmb_shap_beeswarm.png"),
        seed=args.seed,
    )

    # Metrics
    with open(os.path.join(outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Dual HGBR (10+1-feature, Optuna-tuned) — TRAINED ONLY ON ROWS WITH alpha_raw ≤ 0.12\n")
        f.write(f"N={len(df)}  features={feat_cols}\n")
        f.write(f"log_targets={log_targets}\n")
        f.write("Best Optuna params:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nHoldout (same split used in Optuna, train-only fit): "
                f"alpha R2={a_r2_holdout:.3f} MAE={a_mae_holdout:.5f} | "
                f"Pmb R2={p_r2_holdout:.3f} MAE={p_mae_holdout:.5f}\n")
        f.write(f"CV(5-fold, best hyperparams): alpha R2 {np.mean(r2a):.3f}±{np.std(r2a):.3f}; "
                f"Pmb R2 {np.mean(r2p):.3f}±{np.std(r2p):.3f}\n")
        f.write(f"alpha MAE {np.mean(maa):.5f}±{np.std(maa):.5f}; "
                f"Pmb MAE {np.mean(mapb):.5f}±{np.std(mapb):.5f}\n")
        f.write(f"Saved model: {model_path}\n")
        f.write(f"Saved Optuna study: {study_path}\n")

    # Portable inference stub
    stub = os.path.join(outdir, "ersem_infer_stub.py")
    with open(stub, "w", encoding="utf-8") as g:
        g.write(f'''# ersem_infer_stub.py — portable α & PBmax (hourly Bouman units), 10+1-feature Optuna-tuned
import math, numpy as np, pandas as pd, joblib

def daylength_hours(lat_deg, doy):
    lat = math.radians(float(lat_deg))
    g = 2.0*math.pi*(int(doy)-1)/365.0
    dec = (0.006918 - 0.399912*math.cos(g) + 0.070257*math.sin(g)
           - 0.006758*math.cos(2*g) + 0.000907*math.sin(2*g)
           - 0.002697*math.cos(3*g) + 0.00148*math.sin(3*g))

    cwo = -math.tan(lat)*math.tan(dec)
    if cwo >= 1.0:  return 0.0
    if cwo <= -1.0: return 24.0
    return 24.0*math.acos(cwo)/math.pi

def season_harmonics(doy):
    ang = 2.0*math.pi*(int(doy)/365.0)
    return math.sin(ang), math.cos(ang)

def fourier_lon(lon):
    r = math.radians(lon)
    return math.sin(r), math.cos(r)

def features(pack, lat, lon, doy, sst_C, wind_speed_ms, PAR_Wm2, depth_m):
    sin_lon, cos_lon = fourier_lon(lon)
    daylen = daylength_hours(lat, int(doy))
    sin_d, cos_d = season_harmonics(int(doy))

    # hygiene / clipping mirroring training
    sst_C = max(-2.0, min(35.0, float(sst_C)))
    wind_speed_ms = max(0.0, min(25.0, float(wind_speed_ms)))
    PAR_Wm2 = max(0.0, min(1400.0, float(PAR_Wm2)))

    # depth hygiene, mirroring training gate
    depth_m = max(0.0, min(200.0, float(depth_m)))  # <<< ADDED

    sst_x_PAR = float(sst_C) * float(PAR_Wm2)

    row = {{
        "sin_lon": sin_lon,
        "cos_lon": cos_lon,
        "Latitude": float(lat),
        "depth_m": float(depth_m),            # <<< ADDED
        "daylength_h": float(daylen),
        "sin_doy": sin_d,
        "cos_doy": cos_d,
        "sst_C": float(sst_C),
        "wind_speed_ms": float(wind_speed_ms),
        "PAR_Wm2": float(PAR_Wm2),
        "sst_x_PAR": float(sst_x_PAR),
    }}

    X = pd.DataFrame([row])
    for c in pack["feat_cols"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[pack["feat_cols"]]
    X = pack["preprocessor"].transform(X)
    return X

def load(model_path):
    return joblib.load(model_path)

def predict_hourly(pack, lat, lon, doy, sst_C, wind_speed_ms, PAR_Wm2, depth_m):
    X = features(pack, lat, lon, doy, sst_C, wind_speed_ms, PAR_Wm2, depth_m)
    a = pack["alpha_model"].predict(X)[0]
    p = pack["pmb_model"].predict(X)[0]
    if pack["log_targets"]:
        eps = pack["eps"]
        a = 10.0**a - eps
        p = 10.0**p - eps
    return float(a), float(p)
''')

    print("Optuna search complete.")
    print("Saved model:", model_path)
    print("Saved study:", study_path)
    print("Wrote plots, stub, and metrics to:", outdir)

if __name__ == "__main__":
    main()

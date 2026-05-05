#!/usr/bin/env python3
"""
HYB-SF training script
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
Train shelf-focused log-target scaler for GPP:
    z = log(scale), scale = SCOPE_GPP / ERSEM_GPP

Targets:
  - ERSEM baseline GPP integrated to Zeu: gpp_zeu  [mg C m-2 d-1]
  - SCOPE gross PP on ERSEM grid:        scope_pp_gpp [mg C m-2 d-1]

Features (all on ERSEM grid, monthly):
  sst          : SST [degC]
  par          : 24h mean PAR [W m-2]
  wind10m      : 10m wind speed [m s-1]
  sss          : sea surface salinity [PSU]
  lon, lat     : position [deg]
  depth        : bathymetry [m]
  daylength_h  : astronomical daylength [h]
  par_day      : par * daylength_h
  wind2        : wind10m^2
  log1p_depth  : log(1 + depth)
  sst_x_par    : sst * par
  month_sin, month_cos (cyclic month encoding)

Uncertainty handling:
  - scope_pp_unc, if present, is NOT used as a feature.
  - It can be used as:
      * an optional additional quality filter: unc <= unc_max
      * an optional training weight: lower weight for higher uncertainty pixels.

Outputs:
  - model/scaler/bundle (.joblib)
  - JSON metrics report
  - scatter diagnostics: true-vs-predicted scale factor
  - SHAP beeswarm plot
"""

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# SHAP is optional
try:
    import shap
    _HAVE_SHAP = True
except ImportError:
    _HAVE_SHAP = False


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FEATURE_NAMES = [
    "sst",
    "par",
    "wind10m",
    "sss",
    "lon",
    "lat",
    "depth",
    "daylength_h",
    "par_day",
    "wind2",
    "log1p_depth",
    "sst_x_par",
    "month_sin",
    "month_cos",
]

# First 12 columns are continuous numeric features.
# month_sin and month_cos are left unscaled.
N_NUMERIC = 12


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ym_iter(y0, m0, y1, m1):
    """Yield (year, month) from (y0, m0) to (y1, m1), inclusive."""
    y, m = y0, m0
    while (y < y1) or (y == y1 and m <= m1):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def month_features(month: int):
    """Return cyclic sine/cosine encoding for month."""
    ang = 2.0 * math.pi * (month - 1) / 12.0
    return math.sin(ang), math.cos(ang)


def midmonth_doy(year: int, month: int) -> int:
    """Day-of-year for approximately the 15th day of each month."""
    month_lengths = [
        31,
        29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ]
    return sum(month_lengths[: month - 1]) + 15


def daylength_hours(lat_deg: np.ndarray, doy: int) -> np.ndarray:
    """
    Approximate astronomical daylength in hours.

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude in degrees.
    doy : int
        Day of year.

    Returns
    -------
    np.ndarray
        Daylength in hours.
    """
    lat = np.deg2rad(lat_deg)
    delta = 0.409 * np.sin(2.0 * math.pi * (doy - 80) / 365.0)
    cosw = -np.tan(lat) * np.tan(delta)
    cosw = np.clip(cosw, -1.0, 1.0)
    omega = np.arccos(cosw)
    return (24.0 * omega / math.pi).astype(np.float32)


def squeeze_2d(a):
    """Convert a NetCDF variable to a squeezed 2D float32 array."""
    a = np.array(a)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got {a.shape}")
    return a.astype(np.float32, copy=False)


def load_scalar(fp: Path, var: str):
    """Load a named scalar 2D variable from a NetCDF file."""
    with Dataset(fp) as ds:
        if var not in ds.variables:
            raise KeyError(f"{var} not in {fp}")
        return squeeze_2d(ds.variables[var][:])


def load_ersem_gpp(fp: Path):
    """
    Load ERSEM GPP and coordinates.

    Required:
      - gpp_zeu
      - lat or nav_lat
      - lon or nav_lon
    """
    with Dataset(fp) as ds:
        if "gpp_zeu" not in ds.variables:
            raise KeyError(f"gpp_zeu not in {fp}")
        gpp = squeeze_2d(ds.variables["gpp_zeu"][:])

        latvar = "lat" if "lat" in ds.variables else ("nav_lat" if "nav_lat" in ds.variables else None)
        lonvar = "lon" if "lon" in ds.variables else ("nav_lon" if "nav_lon" in ds.variables else None)

        if latvar is None or lonvar is None:
            raise KeyError(f"lat/lon missing in ERSEM file: {fp}")

        lat = squeeze_2d(ds.variables[latvar][:])
        lon = squeeze_2d(ds.variables[lonvar][:])

    return gpp, lat, lon


def load_scope_gpp(fp: Path):
    """
    Load SCOPE GPP and, if available, its uncertainty on the ERSEM grid.

    Returns
    -------
    gpp : 2D float32
        scope_pp_gpp
    unc : 2D float32 or None
        scope_pp_unc, if present.
    """
    with Dataset(fp) as ds:
        if "scope_pp_gpp" not in ds.variables:
            raise KeyError(f"scope_pp_gpp not in {fp}")

        gpp = squeeze_2d(ds["scope_pp_gpp"][:])

        if "scope_pp_unc" in ds.variables:
            unc = squeeze_2d(ds["scope_pp_unc"][:])
        else:
            unc = None

    return gpp, unc


def load_par(fp: Path):
    """Load PAR from one of the accepted variable names."""
    with Dataset(fp) as ds:
        for varname in ("par_on_ersem_24h_wm2", "par_24hmean_wm2", "par_24h_wm2", "par"):
            if varname in ds.variables:
                return squeeze_2d(ds.variables[varname][:])

        raise KeyError(f"PAR variable not found in {fp}")


def load_sst(fp: Path):
    """Load SST [degC]."""
    return load_scalar(fp, "sst_on_ersem_C")


def load_wind(fp: Path):
    """Load 10 m wind speed [m s-1]."""
    return load_scalar(fp, "wind10m_on_ersem_ms")


def load_sss(fp: Path):
    """Load sea surface salinity [PSU]."""
    return load_scalar(fp, "sss_on_ersem_psu")


def metrics(y_true, y_pred):
    """Return MAE, RMSE and R2."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_diag_plot(y_tr, yhat_tr, y_va, yhat_va, y_te, yhat_te, out_png: Path):
    """
    Make a 3-panel scatter diagnostics plot:
    true scale factor vs predicted scale factor for train/validation/test.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for ax, yt, yp, title in zip(
        axes,
        (y_tr, y_va, y_te),
        (yhat_tr, yhat_va, yhat_te),
        ("Train", "Validation", "Test"),
    ):
        if yt.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        ax.scatter(yt, yp, s=2, alpha=0.2)

        lim = [
            np.nanmin([yt.min(), yp.min()]),
            np.nanmax([yt.max(), yp.max()]),
        ]

        if not np.isfinite(lim).all() or lim[0] <= 0:
            lim = [0.1, 10.0]

        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("True scale factor (SCOPE / ERSEM)")
        ax.set_ylabel("Predicted scale factor")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def make_shap_beeswarm_plot(model, X_tr, out_png: Path, rng, max_samples: int = 2000):
    """
    Make a SHAP beeswarm plot showing distribution of per-feature SHAP values.

    Uses a subsample of the training data for speed.
    """
    if not _HAVE_SHAP:
        print("[shap] shap not available; skipping beeswarm plot.")
        return

    n = X_tr.shape[0]

    if n == 0:
        print("[shap] No training samples available for SHAP beeswarm.")
        return

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_used = X_tr[idx]
    else:
        X_used = X_tr

    print(f"[shap] Computing SHAP beeswarm on {X_used.shape[0]} samples...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_used)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_used,
        feature_names=FEATURE_NAMES,
        plot_type="dot",
        show=False,
        max_display=len(FEATURE_NAMES),
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[shap] Saved SHAP beeswarm plot -> {out_png}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train HYB-SF GPP scale-factor model: SCOPE_GPP / ERSEM_GPP."
    )

    # -----------------------------------------------------------------
    # Input roots
    # -----------------------------------------------------------------
    # These are clean relative defaults for the public repository.
    # In normal use, the wrapper script passes machine-specific paths
    # from config/paths/hyb_sf_paths_local.yaml.
    ap.add_argument("--ersem-root", default="data/raw_external/ERSEM")
    ap.add_argument("--scope-root", default="data/raw_external/SCOPE/scope_on_ersem")
    ap.add_argument("--par-root", default="data/raw_external/FEATURES/PAR/unit_wm-2")
    ap.add_argument("--sst-root", default="data/raw_external/FEATURES/SST")
    ap.add_argument("--wind-root", default="data/raw_external/FEATURES/WINDSP10M")
    ap.add_argument("--sss-root", default="data/raw_external/FEATURES/SSS")
    ap.add_argument("--depth-file", default="data/raw_external/FEATURES/STATIC/depth_amm7.nc")

    # -----------------------------------------------------------------
    # File templates
    # -----------------------------------------------------------------
    ap.add_argument("--ersem-tmpl", default="ersem_pp_{YYYY}{MM}.nc")
    ap.add_argument("--scope-tmpl", default="scope_gpp_on_ersem_{YYYY}{MM}.nc")
    ap.add_argument("--par-tmpl", default="par_on_ersem_24h_wm2_{YYYY}{MM}.nc")
    ap.add_argument("--sst-tmpl", default="sst_on_ersem_C_{YYYY}{MM}.nc")
    ap.add_argument("--wind-tmpl", default="wind10m_on_ersem_ms_{YYYY}{MM}.nc")
    ap.add_argument("--sss-tmpl", default="sss_on_ersem_psu_{YYYY}{MM}.nc")

    # -----------------------------------------------------------------
    # Time span and train/validation/test split
    # -----------------------------------------------------------------
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--start-month", type=int, default=3)
    ap.add_argument("--end-year", type=int, default=2015)
    ap.add_argument("--end-month", type=int, default=12)
    ap.add_argument("--train-end-year", type=int, default=2012)
    ap.add_argument("--val-end-year", type=int, default=2014)

    # -----------------------------------------------------------------
    # Filters and weighting
    # -----------------------------------------------------------------
    ap.add_argument("--min-ersem", type=float, default=1e-3)  # mg C m-2 d-1
    ap.add_argument("--scale-min", type=float, default=0.05)
    ap.add_argument("--scale-max", type=float, default=6.0)
    ap.add_argument("--shelf-depth", type=float, default=200.0)
    ap.add_argument("--tail-frac", type=float, default=0.25)
    ap.add_argument("--tail-w", type=float, default=3.0)
    ap.add_argument("--random-seed", type=int, default=42)

    # -----------------------------------------------------------------
    # Uncertainty usage
    # -----------------------------------------------------------------
    ap.add_argument(
        "--use-uncertainty",
        action="store_true",
        help="Use scope_pp_unc as additional training weight and optional filter.",
    )
    ap.add_argument(
        "--unc-max",
        type=float,
        default=None,
        help="Optional hard cutoff in scope_pp_unc; pixels with unc > unc-max are dropped.",
    )

    # -----------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------
    ap.add_argument("--model-out", default="gpp_scale_model_log_shelf.joblib")
    ap.add_argument("--scaler-out", default="gpp_scale_scaler_log_shelf.joblib")
    ap.add_argument("--bundle-out", default="gpp_scale_bundle_log_shelf.joblib")
    ap.add_argument("--report-out", default="gpp_scale_training_report_log_shelf.json")
    ap.add_argument("--diag-plot", default="gpp_scale_diagnostics.png")
    ap.add_argument("--shap-beeswarm-plot", default="gpp_scale_shap_beeswarm.png")

    # Deprecated plotting arguments retained for wrapper/backward compatibility.
    # These are accepted but ignored.
    ap.add_argument("--resid-plot", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--shap-plot", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()
    rng = np.random.default_rng(args.random_seed)

    # Ensure output directories exist.
    # This makes the script safe to run directly, not only through the wrapper.
    for out_file in [
        args.model_out,
        args.scaler_out,
        args.bundle_out,
        args.report_out,
        args.diag_plot,
        args.shap_beeswarm_plot,
    ]:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load static depth
    # -----------------------------------------------------------------
    with Dataset(args.depth_file) as ds_depth:
        if "depth" not in ds_depth.variables:
            raise KeyError(f"depth variable not found in {args.depth_file}")
        depth2d_all = squeeze_2d(ds_depth.variables["depth"][:])

    X_list = []
    z_list = []
    ylin_list = []
    yrs_list = []
    split_tag = []
    unc_list = []

    # -----------------------------------------------------------------
    # Assemble monthly training samples
    # -----------------------------------------------------------------
    for y, m in ym_iter(args.start_year, args.start_month, args.end_year, args.end_month):
        ym = f"{y:04d}{m:02d}"

        paths = {
            "ersem": Path(args.ersem_root) / args.ersem_tmpl.format(YYYY=f"{y:04d}", MM=f"{m:02d}"),
            "scope": Path(args.scope_root) / args.scope_tmpl.format(YYYY=f"{y:04d}", MM=f"{m:02d}"),
            "par": Path(args.par_root) / args.par_tmpl.format(YYYY=f"{y:04d}", MM=f"{m:02d}"),
            "sst": Path(args.sst_root) / args.sst_tmpl.format(YYYY=f"{y:04d}", MM=f"{m:02d}"),
            "wind": Path(args.wind_root) / args.wind_tmpl.format(YYYY=f"{y:04d}", MM=f"{m:02d}"),
            "sss": Path(args.sss_root) / args.sss_tmpl.format(YYYY=f"{y:04d}", MM=f"{m:02d}"),
        }

        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            print(f"[skip] {ym}: missing inputs -> {missing}")
            for name in missing:
                print(f"       {name}: {paths[name]}")
            continue

        try:
            ersem_gpp, lat, lon = load_ersem_gpp(paths["ersem"])
            scope_gpp, scope_unc = load_scope_gpp(paths["scope"])
            par = load_par(paths["par"])
            sst = load_sst(paths["sst"])
            wind = load_wind(paths["wind"])
            sss = load_sss(paths["sss"])
        except Exception as exc:
            print(f"[skip] {ym}: read error -> {exc}")
            continue

        # Shelf mask
        shelf = (
            np.isfinite(depth2d_all)
            & (depth2d_all > 0.0)
            & (depth2d_all <= args.shelf_depth)
        )

        # Valid data mask
        base = (
            shelf
            & np.isfinite(ersem_gpp)
            & (ersem_gpp >= args.min_ersem)
            & np.isfinite(scope_gpp)
            & (scope_gpp > 0.0)
            & np.isfinite(par)
            & np.isfinite(sst)
            & np.isfinite(wind)
            & np.isfinite(sss)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )

        # Optional uncertainty filtering
        if args.use_uncertainty and (scope_unc is not None):
            base = base & np.isfinite(scope_unc)

            if args.unc_max is not None:
                base = base & (scope_unc <= args.unc_max)

        if not base.any():
            print(f"[skip] {ym}: no shelf-valid pixels")
            continue

        # Target ratio and filtering
        ratio = scope_gpp[base] / ersem_gpp[base]

        ok = (
            np.isfinite(ratio)
            & (ratio >= args.scale_min)
            & (ratio <= args.scale_max)
        )

        if not ok.any():
            print(f"[skip] {ym}: no valid ratios after filters")
            continue

        r = ratio[ok].astype(np.float32)      # linear scale factor
        z = np.log(r).astype(np.float32)      # log target

        # Flatten features
        sst_f = sst[base][ok]
        par_f = par[base][ok]
        wind_f = wind[base][ok]
        sss_f = sss[base][ok]
        lon_f = lon[base][ok]
        lat_f = lat[base][ok]
        dep_f = depth2d_all[base][ok]

        # Optional uncertainty vector aligned with accepted samples
        if scope_unc is not None and args.use_uncertainty:
            unc_f = scope_unc[base][ok].astype(np.float32)
        else:
            unc_f = None

        doy = midmonth_doy(y, m)
        dlh = daylength_hours(lat_f, doy)

        par_day = par_f * dlh
        wind2 = wind_f * wind_f
        log1p_depth = np.log1p(dep_f).astype(np.float32)
        sst_x_par = (sst_f * par_f).astype(np.float32)

        sinm, cosm = month_features(m)
        sin_f = np.full_like(sst_f, sinm, dtype=np.float32)
        cos_f = np.full_like(sst_f, cosm, dtype=np.float32)

        X_month = np.column_stack(
            [
                sst_f,
                par_f,
                wind_f,
                sss_f,
                lon_f,
                lat_f,
                dep_f,
                dlh,
                par_day,
                wind2,
                log1p_depth,
                sst_x_par,
                sin_f,
                cos_f,
            ]
        ).astype(np.float32)

        X_list.append(X_month)
        z_list.append(z)
        ylin_list.append(r)
        yrs_list.append(np.full(z.shape, y, dtype=np.int32))

        tag = 0 if y <= args.train_end_year else (1 if y <= args.val_end_year else 2)
        split_tag.append(np.full(z.shape, tag, dtype=np.int8))

        if unc_f is not None:
            unc_list.append(unc_f)

        print(f"[keep] {ym}: N={z.size:,}")

    if not X_list:
        raise RuntimeError("No data assembled after filtering. Check paths, years and filters.")

    X = np.vstack(X_list)
    z = np.concatenate(z_list)
    ylin = np.concatenate(ylin_list)
    years = np.concatenate(yrs_list)
    tags = np.concatenate(split_tag)  # 0=train, 1=validation, 2=test

    # Optional uncertainty vector aligned with rows of X
    if args.use_uncertainty and unc_list:
        unc_all = np.concatenate(unc_list)

        if unc_all.shape[0] != X.shape[0]:
            raise RuntimeError("unc_all and X have inconsistent lengths.")
    else:
        unc_all = None

    m_tr = tags == 0
    m_va = tags == 1
    m_te = tags == 2

    if not m_tr.any():
        raise RuntimeError("No training samples found. Check train_end_year and data period.")

    # -----------------------------------------------------------------
    # Standardise numeric columns using TRAIN only
    # -----------------------------------------------------------------
    scaler = StandardScaler().fit(X[m_tr, :N_NUMERIC])

    X_tr = X[m_tr].copy()
    X_va = X[m_va].copy()
    X_te = X[m_te].copy()

    X_tr[:, :N_NUMERIC] = scaler.transform(X_tr[:, :N_NUMERIC])

    if X_va.size:
        X_va[:, :N_NUMERIC] = scaler.transform(X_va[:, :N_NUMERIC])

    if X_te.size:
        X_te[:, :N_NUMERIC] = scaler.transform(X_te[:, :N_NUMERIC])

    z_tr = z[m_tr]
    z_va = z[m_va]
    z_te = z[m_te]

    y_tr = ylin[m_tr]
    y_va = ylin[m_va]
    y_te = ylin[m_te]

    # -----------------------------------------------------------------
    # Tail weighting in TRAIN on linear ratio
    # -----------------------------------------------------------------
    if y_tr.size > 0:
        thr = np.quantile(y_tr, 1.0 - args.tail_frac)
    else:
        thr = float("nan")

    w_tr = np.ones_like(z_tr, dtype=np.float32)

    if np.isfinite(thr):
        w_tr[y_tr >= thr] = args.tail_w

    # -----------------------------------------------------------------
    # Optional uncertainty weighting in TRAIN only
    # w_unc ~ 1 / (1 + (unc / unc0)^2), clipped to [0.1, 1]
    # -----------------------------------------------------------------
    if unc_all is not None:
        unc_tr = unc_all[m_tr]
        unc0 = np.nanmedian(unc_tr) if np.isfinite(unc_tr).any() else None

        if (unc0 is not None) and (unc0 > 0.0):
            x = np.clip(unc_tr / unc0, 0.0, 10.0)
            w_unc = 1.0 / (1.0 + x * x)
            w_unc = np.clip(w_unc, 0.1, 1.0).astype(np.float32)
            w_tr *= w_unc
            print(f"[unc] Using uncertainty weighting with median unc0 = {unc0:.3g}")
        else:
            print("[unc] Uncertainty present but unc0 invalid; skipping uncertainty weighting.")
    else:
        print("[unc] No uncertainty weighting applied.")

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_depth=6,
        max_iter=400,
        l2_regularization=1e-3,
        min_samples_leaf=100,
        validation_fraction=0.1,
        early_stopping=True,
        random_state=args.random_seed,
    )

    model.fit(X_tr, z_tr, sample_weight=w_tr)

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------
    z_pred_tr = model.predict(X_tr)
    z_pred_va = model.predict(X_va) if z_va.size else np.array([], dtype=np.float32)
    z_pred_te = model.predict(X_te) if z_te.size else np.array([], dtype=np.float32)

    y_pred_tr = np.exp(z_pred_tr)
    y_pred_va = np.exp(z_pred_va) if z_pred_va.size else np.array([], dtype=np.float32)
    y_pred_te = np.exp(z_pred_te) if z_pred_te.size else np.array([], dtype=np.float32)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    rep = {
        "features": FEATURE_NAMES,
        "numeric_scaled": N_NUMERIC,
        "target": "z = log(SCOPE_GPP / ERSEM_gpp_zeu)",
        "data_period": {
            "start": f"{args.start_year:04d}-{args.start_month:02d}",
            "end": f"{args.end_year:04d}-{args.end_month:02d}",
        },
        "split": {
            "train_end_year": args.train_end_year,
            "val_end_year": args.val_end_year,
            "N_train": int(m_tr.sum()),
            "N_val": int(m_va.sum()),
            "N_test": int(m_te.sum()),
        },
        "filters": {
            "shelf_depth_m": args.shelf_depth,
            "min_ersem": args.min_ersem,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
        },
        "tail": {
            "frac": args.tail_frac,
            "weight": args.tail_w,
            "thr_linear": float(thr) if np.isfinite(thr) else None,
        },
        "uncertainty": {
            "used": bool(args.use_uncertainty),
            "unc_max": args.unc_max,
        },
        "metrics": {
            "log": {
                "train": metrics(z_tr, z_pred_tr) if z_tr.size > 0 else None,
                "val": metrics(z_va, z_pred_va) if z_va.size > 0 else None,
                "test": metrics(z_te, z_pred_te) if z_te.size > 0 else None,
            },
            "linear": {
                "train": metrics(y_tr, y_pred_tr) if y_tr.size > 0 else None,
                "val": metrics(y_va, y_pred_va) if y_va.size > 0 else None,
                "test": metrics(y_te, y_pred_te) if y_te.size > 0 else None,
            },
        },
        "model": "HistGradientBoostingRegressor",
        "model_parameters": {
            "loss": "squared_error",
            "learning_rate": 0.06,
            "max_depth": 6,
            "max_iter": 400,
            "l2_regularization": 1e-3,
            "min_samples_leaf": 100,
            "validation_fraction": 0.1,
            "early_stopping": True,
            "random_state": args.random_seed,
        },
    }

    # -----------------------------------------------------------------
    # Save model/scaler/bundle/report
    # -----------------------------------------------------------------
    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURE_NAMES,
            "numeric_scaled": N_NUMERIC,
            "report": rep,
        },
        args.bundle_out,
    )

    with open(args.report_out, "w") as f:
        json.dump(rep, f, indent=2)

    print("\n[report]")
    print(json.dumps(rep["metrics"], indent=2))

    print(
        f"\nSaved:\n"
        f"  {args.model_out}\n"
        f"  {args.scaler_out}\n"
        f"  {args.bundle_out}\n"
        f"  {args.report_out}"
    )

    # -----------------------------------------------------------------
    # Scatter diagnostics only
    # -----------------------------------------------------------------
    make_diag_plot(
        y_tr,
        y_pred_tr,
        y_va,
        y_pred_va,
        y_te,
        y_pred_te,
        Path(args.diag_plot),
    )
    print(f"[diag] Saved scatter diagnostics plot -> {args.diag_plot}")

    # -----------------------------------------------------------------
    # SHAP beeswarm only
    # -----------------------------------------------------------------
    if _HAVE_SHAP:
        try:
            make_shap_beeswarm_plot(model, X_tr, Path(args.shap_beeswarm_plot), rng)
        except Exception as exc:
            print(f"[shap] Failed to compute SHAP beeswarm values: {exc}")
    else:
        print("[shap] shap not installed; skipping SHAP beeswarm plot.")


if __name__ == "__main__":
    main()

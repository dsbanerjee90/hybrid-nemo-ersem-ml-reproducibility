# ersem_infer_stub.py — portable α & PBmax (hourly Bouman units), 10+1-feature Optuna-tuned
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

    row = {
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
    }

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

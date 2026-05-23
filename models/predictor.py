# backend/models/predictor.py
"""
Ensemble Predictor: Prophet + XGBoost residual correction.

Priority (best available):
  1. prophet_model.pkl + xgb_residual.pkl  →  "ensemble"
  2. prophet_model.pkl only                →  "prophet"
  3. ML/model.pkl (legacy sklearn)         →  "legacy"
  4. Nothing                               →  "baseline"

predict_next_days() returns:
  [{"date": "YYYY-MM-DD", "kwh": float,
    "kwh_lower": float, "kwh_upper": float}, ...]
"""

import os, logging, warnings
from datetime import timedelta, date
import numpy as np, pandas as pd, joblib

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logger = logging.getLogger("ecotrack.predictor")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR       = os.path.join(BASE_DIR, "ML")
PROPHET_PATH = os.path.join(ML_DIR, "prophet_model.pkl")
XGB_PATH     = os.path.join(ML_DIR, "xgb_residual.pkl")
MODEL_PATH   = os.path.join(ML_DIR, "model.pkl")          # legacy compat

# ── features (must match train_model.py) ────────────────────────────────────
XGB_FEATURE_COLS = [
    "day_of_week","day_of_year","month","quarter",
    "is_weekend","week_of_year","day_of_month",
    "lag_1","lag_7","lag_14",
    "rolling_mean_7","rolling_mean_14","rolling_std_7",
]

def _calendar(ds_series: pd.Series) -> pd.DataFrame:
    ds = pd.to_datetime(ds_series)
    return pd.DataFrame({
        "day_of_week" : ds.dt.dayofweek,
        "day_of_year" : ds.dt.dayofyear,
        "month"       : ds.dt.month,
        "quarter"     : ds.dt.quarter,
        "is_weekend"  : (ds.dt.dayofweek >= 5).astype(int),
        "week_of_year": ds.dt.isocalendar().week.astype(int).values,
        "day_of_month": ds.dt.day,
    })

def _lag_feats(history: pd.Series, n: int) -> pd.DataFrame:
    """Rolling lag features for n future days, using only actuals (no compounding)."""
    hist = list(history.dropna().values[-30:])
    rows = []
    for _ in range(n):
        l  = len(hist)
        rows.append({
            "lag_1"          : float(hist[-1])  if l>=1  else 0.0,
            "lag_7"          : float(hist[-7])  if l>=7  else (float(hist[0]) if l else 0.0),
            "lag_14"         : float(hist[-14]) if l>=14 else (float(hist[0]) if l else 0.0),
            "rolling_mean_7" : float(np.mean(hist[-7:]))  if l>=1 else 0.0,
            "rolling_mean_14": float(np.mean(hist[-14:])) if l>=1 else 0.0,
            "rolling_std_7"  : float(np.std(hist[-7:]))  if l>1  else 0.0,
        })
    return pd.DataFrame(rows)


# ── Predictor ────────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self):
        self.prophet, self.xgb, self.legacy_model = None, None, None
        self.model_source = "baseline"
        self._load()

    def _load(self):
        p_ok = x_ok = False
        if os.path.exists(PROPHET_PATH):
            try:
                self.prophet = joblib.load(PROPHET_PATH); p_ok = True
                logger.info("Prophet loaded ← %s", PROPHET_PATH)
            except Exception as e: logger.warning("Prophet load failed: %s", e)
        if os.path.exists(XGB_PATH):
            try:
                self.xgb = joblib.load(XGB_PATH); x_ok = True
                logger.info("XGBoost loaded ← %s", XGB_PATH)
            except Exception as e: logger.warning("XGBoost load failed: %s", e)

        if   p_ok and x_ok: self.model_source = "ensemble"
        elif p_ok:           self.model_source = "prophet"
        elif os.path.exists(MODEL_PATH):
            try:
                self.legacy_model = joblib.load(MODEL_PATH)
                self.model_source = "legacy"
                logger.info("Legacy model loaded ← %s", MODEL_PATH)
            except Exception as e: logger.warning("Legacy load failed: %s", e)

        logger.info("Predictor ready — source=%s", self.model_source)

    # ── public API ───────────────────────────────────────────────────────────

    def predict_next_days(self, history_df: pd.DataFrame, days: int = 7) -> list:
        today = date.today()
        if history_df is None or history_df.empty:
            return [{"date":(today+timedelta(days=i)).isoformat(),
                     "kwh":None,"kwh_lower":None,"kwh_upper":None}
                    for i in range(1, days+1)]

        df = history_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        last_date = df["date"].max()

        # ── Prophet / Ensemble ───────────────────────────────────────────────
        if self.prophet is not None:
            try:
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1), periods=days, freq="D")
                future_df = pd.DataFrame({"ds": future_dates})
                fc = self.prophet.predict(future_df)

                point = fc["yhat"].values.clip(min=0)
                lower = fc["yhat_lower"].values.clip(min=0)
                upper = fc["yhat_upper"].values.clip(min=0)

                if self.xgb is not None:
                    cal  = _calendar(future_df["ds"])
                    lags = _lag_feats(df["kwh"], days)
                    X    = pd.concat([cal.reset_index(drop=True),
                                      lags.reset_index(drop=True)], axis=1)
                    corr  = self.xgb.predict(X[XGB_FEATURE_COLS].values)
                    point = (point + corr).clip(min=0)
                    lower = (lower + corr).clip(min=0)
                    upper = (upper + corr).clip(min=0)

                return [
                    {"date":      (last_date + timedelta(days=i)).date().isoformat(),
                     "kwh":       round(float(point[i-1]), 4),
                     "kwh_lower": round(float(lower[i-1]), 4),
                     "kwh_upper": round(float(upper[i-1]), 4)}
                    for i in range(1, days+1)
                ]
            except Exception as e:
                logger.exception("Ensemble prediction failed, falling back: %s", e)

        # ── Legacy sklearn ───────────────────────────────────────────────────
        if self.legacy_model is not None:
            try:
                min_date = df["date"].min()
                offset   = (last_date - min_date).days
                X = [[offset + i] for i in range(1, days+1)]
                y = self.legacy_model.predict(X)
                return [{"date":(last_date+timedelta(days=i)).date().isoformat(),
                         "kwh":round(float(max(0,y[i-1])),4),
                         "kwh_lower":round(float(max(0,y[i-1])),4),
                         "kwh_upper":round(float(max(0,y[i-1])),4)}
                        for i in range(1, days+1)]
            except Exception as e:
                logger.exception("Legacy model failed, falling back: %s", e)

        # ── Baseline ─────────────────────────────────────────────────────────
        return self._baseline(df, last_date, days)

    def _baseline(self, df, last_date, days):
        if len(df) >= 2:
            slope    = float(df["kwh"].iloc[-1] - df["kwh"].iloc[-2])
            last_kwh = float(df["kwh"].iloc[-1])
        else:
            slope    = 0.0
            last_kwh = float(df["kwh"].iloc[-1]) if len(df) == 1 else 0.0
        return [{"date":(last_date+timedelta(days=i)).date().isoformat(),
                 "kwh":round(max(0.0, last_kwh+slope*i),4),
                 "kwh_lower":round(max(0.0, last_kwh+slope*i),4),
                 "kwh_upper":round(max(0.0, last_kwh+slope*i),4)}
                for i in range(1, days+1)]

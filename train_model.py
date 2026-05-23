# backend/train_model.py
"""
Ensemble trainer: Prophet (trend + seasonality) + XGBoost (residual correction).

Pipeline
────────
1. Load all EnergyUsage records from DB → aggregate to daily totals.
2. Fit Prophet on the full series   → captures trend & weekly seasonality.
3. Compute residuals = actual - Prophet in-sample fit.
4. Fit XGBoost on those residuals with calendar + lag features.
5. Save both models  →  ML/prophet_model.pkl  and  ML/xgb_residual.pkl.

At inference (predictor.py):
  final_kwh = prophet_forecast + xgb_residual_correction
"""

import os, math, warnings, logging
import pandas as pd, numpy as np, joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from prophet import Prophet

from app import app, EnergyUsage
from models.predictor import PROPHET_PATH, XGB_PATH, XGB_FEATURE_COLS


# ── helpers ──────────────────────────────────────────────────────────────────

def section(t): print(f"\n{'═'*70}\n  {t}\n{'═'*70}")
def ok(m):      print(f"  ✔  {m}")
def info(m):    print(f"     {m}")


# ── Step 1 — load ─────────────────────────────────────────────────────────────

def load_daily_df() -> pd.DataFrame:
    """Pull every EnergyUsage row, aggregate to daily kWh totals (all users)."""
    section("STEP 1 — Loading data from database")
    with app.app_context():
        rows = EnergyUsage.query.order_by(EnergyUsage.date.asc()).all()
    if not rows:
        raise RuntimeError("No EnergyUsage records found. Add data first.")

    raw = pd.DataFrame([{"date": r.date, "kwh": float(r.kwh)} for r in rows])
    raw["date"] = pd.to_datetime(raw["date"])
    daily = (
        raw.groupby("date", as_index=False)["kwh"]
           .sum()
           .rename(columns={"date": "ds", "kwh": "y"})
           .sort_values("ds").reset_index(drop=True)
    )
    ok(f"Loaded {len(rows):,} records → {len(daily):,} unique daily totals")
    info(f"Range : {daily['ds'].min().date()} → {daily['ds'].max().date()}")
    info(f"kWh   : min={daily['y'].min():.2f}  max={daily['y'].max():.2f}  mean={daily['y'].mean():.2f}")
    return daily


# ── Step 2 — feature builder (shared with predictor.py) ───────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + lag features. 'y' column used for lags when present."""
    d = df.copy().reset_index(drop=True)
    d["ds"] = pd.to_datetime(d["ds"])
    d["day_of_week"]  = d["ds"].dt.dayofweek
    d["day_of_year"]  = d["ds"].dt.dayofyear
    d["month"]        = d["ds"].dt.month
    d["quarter"]      = d["ds"].dt.quarter
    d["is_weekend"]   = (d["ds"].dt.dayofweek >= 5).astype(int)
    d["week_of_year"] = d["ds"].dt.isocalendar().week.astype(int).values
    d["day_of_month"] = d["ds"].dt.day
    kwh = "y" if "y" in d.columns else None
    if kwh:
        d["lag_1"]           = d[kwh].shift(1)
        d["lag_7"]           = d[kwh].shift(7)
        d["lag_14"]          = d[kwh].shift(14)
        d["rolling_mean_7"]  = d[kwh].shift(1).rolling(7,  min_periods=1).mean()
        d["rolling_mean_14"] = d[kwh].shift(1).rolling(14, min_periods=1).mean()
        d["rolling_std_7"]   = d[kwh].shift(1).rolling(7,  min_periods=1).std().fillna(0)
    else:
        for c in ["lag_1","lag_7","lag_14","rolling_mean_7","rolling_mean_14","rolling_std_7"]:
            d[c] = 0.0
    return d.fillna(0)


# ── Step 3 — train Prophet ────────────────────────────────────────────────────

def train_prophet(daily: pd.DataFrame):
    section("STEP 2 — Fitting Prophet  (trend + weekly seasonality)")
    n = len(daily)
    model = Prophet(
        yearly_seasonality      = (n >= 365),
        weekly_seasonality      = True,
        daily_seasonality       = False,
        seasonality_mode        = "multiplicative",
        changepoint_prior_scale = 0.05,
        interval_width          = 0.80,
    )
    model.fit(daily)

    # In-sample fitted values needed for residual computation
    fitted = model.predict(daily[["ds"]])[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fitted["yhat"] = fitted["yhat"].clip(lower=0)

    holdout = max(7, n // 10)
    merged  = daily.merge(fitted, on="ds").tail(holdout)
    mae     = mean_absolute_error(merged["y"], merged["yhat"])
    rmse    = math.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
    ok(f"Prophet fitted on {n} days (hold-out last {holdout})")
    info(f"MAE={mae:.3f}  RMSE={rmse:.3f}")
    return model, fitted


# ── Step 4 — train XGBoost on residuals ───────────────────────────────────────

def train_xgb_residual(daily: pd.DataFrame, prophet_fitted: pd.DataFrame) -> XGBRegressor:
    section("STEP 3 — Fitting XGBoost on Prophet residuals")
    merged = daily.merge(prophet_fitted[["ds","yhat"]], on="ds")
    merged["residual"] = merged["y"] - merged["yhat"]

    feats = build_features(merged.rename(columns={"y":"y"}))
    X, y  = feats[XGB_FEATURE_COLS].values, merged["residual"].values

    n       = len(X)
    holdout = max(14, n // 5) if n > 28 else max(1, n // 5)
    X_tr, X_te = X[:-holdout], X[-holdout:]
    y_tr, y_te = y[:-holdout], y[-holdout:]

    xgb = XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=4,
        subsample=0.8,    colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1,    reg_lambda=1.0,       random_state=42,
        n_jobs=-1,        verbosity=0,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = xgb.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = math.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    ok(f"XGBoost trained (train={len(X_tr)}, test={len(X_te)})")
    info(f"Residual MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    top5 = sorted(zip(XGB_FEATURE_COLS, xgb.feature_importances_), key=lambda x:-x[1])[:5]
    info("Top features: " + "  ".join(f"{f}={v:.3f}" for f,v in top5))
    return xgb


# ── Step 5 — combined evaluation ──────────────────────────────────────────────

def evaluate_ensemble(daily, prophet, xgb):
    section("STEP 4 — Ensemble evaluation (Prophet + XGBoost combined)")
    n       = len(daily)
    holdout = max(14, n // 5) if n > 28 else max(1, n // 5)
    test    = daily.tail(holdout).copy()

    future        = prophet.make_future_dataframe(periods=0)
    prophet_pred  = prophet.predict(future)[["ds","yhat"]].copy()
    prophet_pred["yhat"] = prophet_pred["yhat"].clip(lower=0)
    test_m        = test.merge(prophet_pred, on="ds", how="left")

    full_feat     = build_features(daily)
    xgb_resid     = xgb.predict(full_feat.tail(holdout)[XGB_FEATURE_COLS].values)
    ensemble      = (test_m["yhat"].values + xgb_resid).clip(min=0)
    actual        = test_m["y"].values

    mae_e  = mean_absolute_error(actual, ensemble)
    rmse_e = math.sqrt(mean_squared_error(actual, ensemble))
    r2_e   = r2_score(actual, ensemble)
    mae_p  = mean_absolute_error(actual, test_m["yhat"].values.clip(min=0))
    delta  = (mae_p - mae_e) / mae_p * 100 if mae_p > 0 else 0

    ok(f"Ensemble on last {holdout} days:")
    info(f"MAE  ensemble={mae_e:.3f}   Prophet-only={mae_p:.3f}   improvement={delta:.1f}%")
    info(f"RMSE={rmse_e:.3f}   R²={r2_e:.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    section("ECOTRACK ENSEMBLE TRAINER  (Prophet + XGBoost)")
    daily = load_daily_df()
    if len(daily) < 14:
        print("\n⚠  < 14 daily records — results may be unreliable.")

    prophet_model, prophet_fitted = train_prophet(daily)
    xgb_model = train_xgb_residual(daily, prophet_fitted)
    if len(daily) >= 28:
        evaluate_ensemble(daily, prophet_model, xgb_model)

    section("STEP 5 — Saving models")
    os.makedirs(os.path.dirname(PROPHET_PATH), exist_ok=True)
    joblib.dump(prophet_model, PROPHET_PATH); ok(f"Prophet  → {PROPHET_PATH}")
    joblib.dump(xgb_model,    XGB_PATH);     ok(f"XGBoost  → {XGB_PATH}")
    section("TRAINING COMPLETE ✔")

if __name__ == "__main__":
    main()

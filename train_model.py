# backend/train_model.py

import os
import math
from datetime import datetime

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# import your Flask app, db, and model
from app import app, EnergyUsage
from models.predictor import MODEL_PATH


def log_header(title: str):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def load_history_df():
    """
    Load all EnergyUsage records into a pandas DataFrame.
    Columns: ['date', 'kwh', 'company', 'user_id']
    """
    log_header("STEP 1: Loading data from database")

    with app.app_context():
        rows = (
            EnergyUsage.query
            .order_by(EnergyUsage.date.asc())
            .all()
        )

        if not rows:
            raise RuntimeError("No energy usage data found in DB. Add some records first.")

        data = [
            {
                "date": r.date,   # date object
                "kwh": float(r.kwh),
                "company": r.company,
                "user_id": r.user_id,
            }
            for r in rows
        ]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Basic dataset info
    print(f"Total records loaded : {len(df)}")
    print(f"Unique users         : {df['user_id'].nunique()}")
    print(f"Unique companies     : {df['company'].nunique()}")
    print(f"Date range           : {df['date'].min().date()} -> {df['date'].max().date()}")

    # Per-company counts
    print("\nRecords per company:")
    print(df.groupby("company")["kwh"].count().to_string())

    # Small preview of raw data
    print("\nSample of raw data (first 5 rows):")
    print(df.head().to_string(index=False))

    return df


def build_features_simple_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, Predictor-compatible feature:
    - day_index = number of days since the first date in the dataset.

    This matches the logic in your current Predictor, which uses a single
    time-offset feature to call model.predict(X).
    """
    log_header("STEP 2: Building features (day_index)")

    df = df.sort_values("date").copy()
    min_date = df["date"].min()
    df["day_index"] = (df["date"] - min_date).dt.days

    # Feature summary
    print("Feature: day_index (days since first record)")
    print(f"  min day_index : {df['day_index'].min()}")
    print(f"  max day_index : {df['day_index'].max()}")
    print(f"  mean day_index: {df['day_index'].mean():.2f}")

    print("\nTarget: kwh")
    print(f"  min kwh : {df['kwh'].min():.4f}")
    print(f"  max kwh : {df['kwh'].max():.4f}")
    print(f"  mean kwh: {df['kwh'].mean():.4f}")

    # Preview with features
    print("\nSample of data with features (first 5 rows):")
    print(df[["date", "day_index", "kwh", "company"]].head().to_string(index=False))

    return df


def train_time_model(df: pd.DataFrame):
    """
    Train a RandomForestRegressor using only the day_index feature.
    This is fully compatible with your current Predictor.predict_next_days,
    which constructs X = [[offset]].
    """
    log_header("STEP 3: Training model (RandomForestRegressor on day_index)")

    df_feat = build_features_simple_time(df)

    # Feature and target
    X = df_feat[["day_index"]].values
    y = df_feat["kwh"].values

    n_samples = len(df_feat)
    print(f"\nTotal samples available for training: {n_samples}")

    if n_samples < 10:
        print("⚠️  Very few samples (<10). Model will train, but predictions may be unstable.")

    # Simple time-based split: last 7 points as test (if enough data)
    if n_samples > 14:
        train_size = n_samples - 7
        split_desc = "last 7 samples as test set"
    else:
        train_size = int(n_samples * 0.8)
        split_desc = "80/20 train/test split (not enough for 7-sample holdout)"

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Train/Test split: {split_desc}")
    print(f"  Train size: {len(X_train)}")
    print(f"  Test size : {len(X_test)}")

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate if we have a test set
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        log_header("STEP 4: Evaluation on hold-out set")
        print(f"  MAE : {mae:.4f} kWh")
        print(f"  RMSE: {rmse:.4f} kWh")
        print(f"  R^2 : {r2:.4f}")

        # Optional: show comparison table for last few test points
        comp_df = pd.DataFrame({
            "day_index": df_feat["day_index"].iloc[train_size:],
            "kwh_actual": y_test,
            "kwh_pred": y_pred,
        })
        print("\nSample of actual vs predicted on test set:")
        print(comp_df.head(10).to_string(index=False))
    else:
        print("Not enough samples for a test set; trained on all data.")

    return model


def main():
    log_header("TRAINING SCRIPT STARTED")

    print("Loading history from database...")
    df = load_history_df()

    print("\nTraining model (RandomForestRegressor on day_index)...")
    model = train_time_model(df)

    # Ensure ML directory exists (MODEL_PATH comes from models.predictor)
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    log_header("STEP 5: Saving model")
    print(f"Saving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    log_header("TRAINING COMPLETE")
    print("Your Predictor will now use this saved_model instead of the baseline.")


if __name__ == "__main__":
    main()

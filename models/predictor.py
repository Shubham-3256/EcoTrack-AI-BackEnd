"""
Predictor: tries to load a trained scikit-learn model from ../ML/model.pkl
If not present, fall back to a naive baseline predictor (mean or linear extrapolation).
"""

import os
import joblib
from datetime import timedelta, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ML", "model.pkl")

class Predictor:
    def __init__(self):
        self.model = None
        self.model_source = "baseline"
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.model_source = "saved_model"
            except Exception as e:
                print("Failed loading model.pkl:", e)
                self.model = None

    def predict_next_days(self, history_df, days=7):
        """
        history_df: pandas DataFrame with columns ['date', 'kwh'] OR None
        returns list of dicts [{"date": "YYYY-MM-DD", "kwh": value}, ...]
        """
        # simple baseline: if we have history, compute average daily change and extrapolate
        preds = []
        today = date.today()
        if history_df is None or history_df.empty:
            # no history: return constant default
            for i in range(1, days + 1):
                preds.append({"date": (today + timedelta(days=i)).isoformat(), "kwh": None})
            return preds

        # ensure sorted
        history_df = history_df.sort_values("date")
        # if model exists, try to use it
        if self.model is not None:
            try:
                # expect model.predict to accept a 2D array of day offsets or timestamps
                import pandas as pd
                last_date = history_df["date"].max()
                start_offset = (last_date - history_df["date"].min()).days + 1
                X = [[start_offset + i] for i in range(1, days + 1)]
                y_pred = self.model.predict(X)
                for i, val in enumerate(y_pred, start=1):
                    preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": float(val)})
                return preds
            except Exception as e:
                print("Model predict failed, falling back to baseline:", e)

        # Baseline: linear extrapolation using last two points if available
        if len(history_df) >= 2:
            last = history_df.iloc[-1]
            prev = history_df.iloc[-2]
            slope = (last["kwh"] - prev["kwh"])  # per day
            last_date = last["date"]
            for i in range(1, days + 1):
                predicted = float(max(0.0, last["kwh"] + slope * i))
                preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": predicted})
            return preds

        # If only one data point, return same value
        last = history_df.iloc[-1]
        last_date = last["date"]
        for i in range(1, days + 1):
            preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": float(last["kwh"])})
        return preds

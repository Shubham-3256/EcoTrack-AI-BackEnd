# backend/app.py
import os
import logging
import time
from datetime import datetime
from functools import wraps
# near other imports in app.py
import bcrypt as _bcrypt_lib   # used only for legacy bcrypt verification

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import jwt

# use werkzeug for password hashing (avoids passlib/bcrypt dependency issues)
from werkzeug.security import generate_password_hash, check_password_hash

# local imports that should exist
from database.db import db
from utils.emission_calc import calculate_emission_from_kwh

# ------------------------------------------------------------
# App & basic config
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ecotrack")

app = Flask(__name__)
CORS(app)  # dev only; tighten in production

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)

DB_PATH = os.environ.get("DATABASE_URL") or f"sqlite:///{os.path.join(INSTANCE_DIR, 'ecotrack.db')}"
app.config["SQLALCHEMY_DATABASE_URI"] = DB_PATH
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

JWT_SECRET = os.environ.get("JWT_SECRET") or "jw_kjdashlkjfh@325425"
JWT_ALGORITHM = "HS256"
JWT_EXP_SECONDS = 60 * 60 * 24 * 7  # token valid 7 days

# initialize SQLAlchemy with the app
db.init_app(app)
# ------------------------------------------------------------
# DB auto-init for deployment (Render, etc.)
# ------------------------------------------------------------
with app.app_context():
    try:
        from sqlalchemy import inspect

        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        if "user" not in tables or "energy_usage" not in tables:
            logger.info("Initializing database tables on startup...")
            db.create_all()
            tables = inspector.get_table_names()
            logger.info("DB tables after init: %s", tables)
        else:
            logger.info("DB already initialized with tables: %s", tables)
    except Exception as e:
        logger.exception("DB auto-init failed: %s", e)

# ------------------------------------------------------------
# JWT helpers
# ------------------------------------------------------------
def create_jwt_for_user(user):
    payload = {
        "sub": user.id,
        "email": user.email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXP_SECONDS,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_jwt(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("token_expired")
    except Exception:
        raise ValueError("invalid_token")

def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", None)
        if not auth:
            return jsonify({"error": "missing_authorization"}), 401
        parts = auth.split()
        if parts[0].lower() != "bearer" or len(parts) != 2:
            return jsonify({"error": "invalid_authorization_header"}), 401
        token = parts[1]
        try:
            payload = decode_jwt(token)
        except ValueError as e:
            msg = str(e)
            return jsonify({"error": msg}), 401
        # load user
        user = User.query.get(payload["sub"])
        if not user:
            return jsonify({"error": "user_not_found"}), 401
        # attach current user to flask.g
        g.current_user = user
        return f(*args, **kwargs)
    return decorated

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw_password):
        # use werkzeug's PBKDF2-based hash (no 72-byte bcrypt limit)
        # default method 'pbkdf2:sha256' with salt and iterations
        self.password_hash = generate_password_hash(raw_password)

    def verify_password(self, raw_password):
        """
        Try verifying in this order:
         1. Werkzeug (new format)
         2. Legacy bcrypt ($2b$ / $2a$)
        If legacy bcrypt verification succeeds, rehash to Werkzeug and persist.
        """
        # 1) Try Werkzeug check first
        try:
            if check_password_hash(self.password_hash, raw_password):
                return True
        except Exception:
            # fall through to bcrypt attempt
            pass

        # 2) If hash looks like bcrypt ($2a$, $2b$), try bcrypt library
        try:
            ph = (self.password_hash or "").encode("utf-8")
            pw = raw_password.encode("utf-8")
            if ph.startswith(b"$2a$") or ph.startswith(b"$2b$") or ph.startswith(b"$2y$"):
                if _bcrypt_lib.checkpw(pw, ph):
                    # migrate: rehash with Werkzeug (pbkdf2) for future logins
                    try:
                        self.password_hash = generate_password_hash(raw_password)
                        db.session.add(self)
                        db.session.commit()
                    except Exception:
                        # if rehash save fails — ignore silently but return True
                        logger.exception("Failed to rehash legacy bcrypt user")
                    return True
        except Exception:
            pass

        # failed both
        return False

    def to_public(self):
        return {"id": self.id, "email": self.email, "name": self.name, "created_at": self.created_at.isoformat()}

class EnergyUsage(db.Model):
    __tablename__ = "energy_usage"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    date = db.Column(db.Date, nullable=False)
    kwh = db.Column(db.Float, nullable=False)
    notes = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("energy_usage", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "company": self.company,
            "date": self.date.isoformat(),
            "kwh": self.kwh,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }

# ------------------------------------------------------------
# Predictor: lazy-load and resilient
# ------------------------------------------------------------
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is not None:
        return _predictor_instance

    try:
        from models.predictor import Predictor  # local module
    except Exception as e:
        logger.warning("Could not import models.predictor: %s", e)
        _predictor_instance = None
        return None

    try:
        _predictor_instance = Predictor()
        logger.info("Predictor initialized. model_source=%s", getattr(_predictor_instance, "model_source", None))
    except Exception as e:
        logger.warning("Predictor instantiation failed: %s", e)
        _predictor_instance = None

    return _predictor_instance

# ------------------------------------------------------------
# Auto-train ML model on startup (if missing)
# ------------------------------------------------------------
def auto_train_model():
    """
    If ML/model.pkl does not exist, run train_model.py once to create it.
    This runs in a separate process to avoid circular imports.
    """
    try:
        from models.predictor import MODEL_PATH  # only need the path, not Predictor itself
    except Exception as e:
        logger.exception("Could not import MODEL_PATH from models.predictor: %s", e)
        return

    if os.path.exists(MODEL_PATH):
        logger.info("Model file already exists at %s; skipping auto-training.", MODEL_PATH)
        return

    logger.info("No model found at %s; running train_model.py to create one...", MODEL_PATH)

    try:
        import subprocess
        import sys

        train_script = os.path.join(BASE_DIR, "train_model.py")
        # Run: python train_model.py
        subprocess.run([sys.executable, train_script], check=True)
        logger.info("Auto-training completed successfully.")
    except Exception as e:
        logger.exception("Auto-training failed: %s", e)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def index():
    return jsonify({"message": "EcoTrack AI backend — available endpoints: /health, /history, /save-energy-usage, /calculate-emission, /predict-trend, /companies, /auth/register, /auth/login, /auth/me"}), 200

@app.route("/health", methods=["GET"])
def health():
    try:
        with app.app_context():
            engine_available = bool(db.engine)
            inspector = __import__("sqlalchemy").inspect(db.engine)
            tables = inspector.get_table_names()
    except Exception as e:
        logger.exception("Health check DB error")
        return jsonify({"status": "error", "db": False, "error": str(e)}), 500

    return jsonify({"status": "ok", "db": True, "tables": tables}), 200

@app.route("/save-energy-usage", methods=["POST"])
@jwt_required
def save_energy_usage():
    data = request.get_json() or {}
    company = data.get("company")
    date_str = data.get("date")
    kwh = data.get("kwh")

    if not (company and date_str and (kwh is not None)):
        return jsonify({"error": "company, date, and kwh are required"}), 400

    try:
        date_obj = datetime.fromisoformat(date_str).date()
    except Exception:
        return jsonify({"error": "date must be ISO format YYYY-MM-DD"}), 400

    try:
        kwh_val = float(kwh)
    except Exception:
        return jsonify({"error": "kwh must be numeric"}), 400

    try:
        record = EnergyUsage(user_id=g.current_user.id, company=company, date=date_obj, kwh=kwh_val, notes=data.get("notes"))
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        logger.exception("Failed to save energy usage")
        db.session.rollback()
        return jsonify({"error": "failed to save record", "details": str(e)}), 500

    return jsonify({"saved": record.to_dict()}), 201

@app.route("/calculate-emission", methods=["POST"])
def calculate_emission():
    data = request.get_json() or {}
    if "kwh" not in data:
        return jsonify({"error": "kwh required"}), 400

    try:
        kwh_val = float(data.get("kwh"))
    except Exception:
        return jsonify({"error": "kwh must be numeric"}), 400

    factor = data.get("emission_factor")
    try:
        co2_kg = calculate_emission_from_kwh(kwh_val, emission_factor=factor)
    except Exception as e:
        logger.exception("Emission calc failed")
        return jsonify({"error": "emission calculation failed", "details": str(e)}), 500

    return jsonify({"kwh": kwh_val, "emission_factor": float(co2_kg / kwh_val) if kwh_val != 0 else None, "co2_kg": co2_kg}), 200

@app.route("/predict-trend", methods=["POST"])
@jwt_required
def predict_trend():
    data = request.get_json() or {}
    company = data.get("company")
    try:
        days = int(data.get("days", 7))
    except Exception:
        days = 7

    query = EnergyUsage.query.filter_by(user_id=g.current_user.id)
    if company:
        query = query.filter_by(company=company)
    history = query.order_by(EnergyUsage.date.asc()).all()

    history_df = None
    if history:
        try:
            import pandas as pd
            history_df = pd.DataFrame([{"date": r.date, "kwh": r.kwh} for r in history])
        except Exception as e:
            logger.warning("Could not build history DataFrame: %s", e)
            history_df = None

    predictor = get_predictor()
    if predictor is None:
        preds = []
        from datetime import timedelta, date
        today = date.today()
        if history_df is not None and not history_df.empty:
            history_df = history_df.sort_values("date")
            if len(history_df) >= 2:
                last = history_df.iloc[-1]
                prev = history_df.iloc[-2]
                slope = float(last["kwh"] - prev["kwh"])
                last_date = last["date"]
                for i in range(1, days + 1):
                    predicted = float(max(0.0, last["kwh"] + slope * i))
                    preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": predicted})
            else:
                last = history_df.iloc[-1]
                last_date = last["date"]
                for i in range(1, days + 1):
                    preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": float(last["kwh"])})
        else:
            for i in range(1, days + 1):
                preds.append({"date": (today + timedelta(days=i)).isoformat(), "kwh": None})
        return jsonify({"predictions": preds, "model_type": "none"}), 200

    try:
        preds = predictor.predict_next_days(history_df, days=days)
        return jsonify({"predictions": preds, "model_type": predictor.model_source}), 200
    except Exception as e:
        logger.exception("Predictor failed, falling back to baseline: %s", e)
        from datetime import timedelta, date
        preds = []
        today = date.today()
        if history_df is not None and not history_df.empty:
            history_df = history_df.sort_values("date")
            if len(history_df) >= 2:
                last = history_df.iloc[-1]
                prev = history_df.iloc[-2]
                slope = float(last["kwh"] - prev["kwh"])
                last_date = last["date"]
                for i in range(1, days + 1):
                    predicted = float(max(0.0, last["kwh"] + slope * i))
                    preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": predicted})
            else:
                last = history_df.iloc[-1]
                last_date = last["date"]
                for i in range(1, days + 1):
                    preds.append({"date": (last_date + timedelta(days=i)).isoformat(), "kwh": float(last["kwh"])})
        else:
            for i in range(1, days + 1):
                preds.append({"date": (today + timedelta(days=i)).isoformat(), "kwh": None})
        return jsonify({"predictions": preds, "model_type": "baseline"}), 200

@app.route("/history", methods=["GET"])
@jwt_required
def get_history():
    company = request.args.get("company")
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    query = EnergyUsage.query.filter_by(user_id=g.current_user.id)
    if company:
        query = query.filter_by(company=company)
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date).date()
            query = query.filter(EnergyUsage.date >= from_dt)
        except Exception:
            return jsonify({"error": "invalid from date, expected YYYY-MM-DD"}), 400
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date).date()
            query = query.filter(EnergyUsage.date <= to_dt)
        except Exception:
            return jsonify({"error": "invalid to date, expected YYYY-MM-DD"}), 400

    rows = query.order_by(EnergyUsage.date.asc()).all()
    return jsonify([r.to_dict() for r in rows]), 200

@app.route("/companies", methods=["GET"])
@jwt_required
def get_companies():
    try:
        names = (
            db.session.query(EnergyUsage.company)
            .filter(EnergyUsage.user_id == g.current_user.id)
            .distinct()
            .order_by(EnergyUsage.company.asc())
            .all()
        )
        companies = [n[0] for n in names]
        return jsonify(companies), 200
    except Exception as e:
        logger.exception("Failed to fetch companies")
        return jsonify({"error": "failed_to_fetch_companies", "details": str(e)}), 500

@app.route("/delete-energy-usage", methods=["POST"])
@jwt_required
def delete_energy_usage():
    data = request.get_json() or {}
    rec_id = data.get("id")
    if not rec_id:
        return jsonify({"error":"id_required"}), 400
    rec = EnergyUsage.query.filter_by(id=rec_id, user_id=g.current_user.id).first()
    if not rec:
        return jsonify({"error":"not_found"}), 404
    db.session.delete(rec)
    db.session.commit()
    return jsonify({"deleted": rec_id}), 200

@app.route("/update-energy-usage", methods=["POST"])
@jwt_required
def update_energy_usage():
    data = request.get_json() or {}
    rec_id = data.get("id")
    if not rec_id:
        return jsonify({"error":"id_required"}), 400
    rec = EnergyUsage.query.filter_by(id=rec_id, user_id=g.current_user.id).first()
    if not rec:
        return jsonify({"error":"not_found"}), 404
    # allow updating fields
    rec.company = data.get("company", rec.company)
    if "date" in data:
        rec.date = datetime.fromisoformat(data["date"]).date()
    if "kwh" in data:
        rec.kwh = float(data["kwh"])
    rec.notes = data.get("notes", rec.notes)
    db.session.commit()
    return jsonify({"updated": rec.to_dict()}), 200


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password")
    name = data.get("name")

    if not email or not password:
        return jsonify({"error": "email_and_password_required"}), 400

    # Enforce minimum password length (server-side)
    if len(password) < 8:
        return jsonify({"error": "password_too_short", "min_length": 8}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email_already_registered"}), 400
    user = User(email=email, name=name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    token = create_jwt_for_user(user)
    return jsonify({"token": token, "user": user.to_public()}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "email_and_password_required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.verify_password(password):
        return jsonify({"error": "invalid_credentials"}), 401

    token = create_jwt_for_user(user)
    return jsonify({"token": token, "user": user.to_public()}), 200

@app.route("/auth/me", methods=["GET"])
@jwt_required
def me():
    user = g.current_user
    return jsonify({"user": user.to_public()}), 200

# ------------------------------------------------------------
# CLI helper to initialize DB
# ------------------------------------------------------------
@app.cli.command("init-db")
def init_db():
    logger.info("Initializing DB at %s", app.config.get("SQLALCHEMY_DATABASE_URI"))
    with app.app_context():
        from database.db import db as db_instance
        db_instance.create_all()
        inspector = __import__("sqlalchemy").inspect(db_instance.engine)
        tables = inspector.get_table_names()
        logger.info("Tables now: %s", tables)
    logger.info("Database initialized.")

# ------------------------------------------------------------
# Run (dev)
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1) Auto-train model if needed
    auto_train_model()

    # 2) Start the server
    logger.info(
        "Starting EcoTrack backend on %s (DB=%s)",
        "0.0.0.0:5000",
        app.config.get("SQLALCHEMY_DATABASE_URI"),
    )
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

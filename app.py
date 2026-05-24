# backend/app.py
import os, csv, io, logging, time, subprocess, sys
from datetime import datetime, date, timedelta
from functools import wraps

import bcrypt as _bcrypt_lib
import pandas as pd
from flask import Flask, request, jsonify, g, make_response
from flask_cors import CORS
import random

import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
from utils.emission_calc import calculate_emission_from_kwh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ecotrack")

app = Flask(__name__)


# ── Flask Mail ─────────────────────────────

# app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")
# app.config["MAIL_PORT"] = int(os.environ.get("MAIL_PORT", 587))
# app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS", "True") == "True"
# app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
# app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")

# ── CORS ─────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000"
    ).split(",")
]

CORS(
    app,
    origins=ALLOWED_ORIGINS,
    supports_credentials=True,
)

# ── Database ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DATABASE_URL")
if not DB_PATH:
    raise RuntimeError("DATABASE_URL environment variable is not set")
# Fix Render's legacy postgres:// prefix → postgresql://
if DB_PATH.startswith("postgres://"):
    DB_PATH = "postgresql://" + DB_PATH[len("postgres://"):]

app.config["SQLALCHEMY_DATABASE_URI"] = DB_PATH
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# ── JWT ───────────────────────────────────────────────────────────────────────
JWT_SECRET      = os.environ.get("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM   = "HS256"
JWT_EXP_SECONDS = 60 * 60 * 24 * 7

def create_jwt_for_user(user):
    payload = {"sub": user.id, "email": user.email,
               "iat": int(time.time()), "exp": int(time.time()) + JWT_EXP_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("token_expired")
    except Exception:
        raise ValueError("invalid_token")

def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        parts = auth.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"error": "missing_authorization"}), 401
        try:
            payload = decode_jwt(parts[1])
        except ValueError as e:
            return jsonify({"error": str(e)}), 401
        user = User.query.get(payload["sub"])
        if not user:
            return jsonify({"error": "user_not_found"}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated

# ── Models ────────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "user"
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name          = db.Column(db.String(200), nullable=True)
    # alert settings
    alert_threshold_kwh = db.Column(db.Float, nullable=True)
    alert_email_enabled = db.Column(db.Boolean, default=False)
    # ai tips cache
    tips_cache      = db.Column(db.Text, nullable=True)
    tips_cache_date = db.Column(db.Date, nullable=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw):
        self.password_hash = generate_password_hash(raw)

    def verify_password(self, raw):
        try:
            if check_password_hash(self.password_hash, raw):
                return True
        except Exception:
            pass
        try:
            ph = (self.password_hash or "").encode("utf-8")
            pw = raw.encode("utf-8")
            if ph.startswith((b"$2a$", b"$2b$", b"$2y$")):
                if _bcrypt_lib.checkpw(pw, ph):
                    self.password_hash = generate_password_hash(raw)
                    try:
                        db.session.commit()
                    except Exception:
                        pass
                    return True
        except Exception:
            pass
        return False

    def to_public(self):
        return {
            "id": self.id, "email": self.email, "name": self.name,
            "alert_threshold_kwh": self.alert_threshold_kwh,
            "alert_email_enabled": self.alert_email_enabled,
            "created_at": self.created_at.isoformat(),
        }
@app.route("/auth/verify-reset-otp", methods=["POST"])
def verify_reset_otp():
    data = request.get_json() or {}

    email = (data.get("email") or "").strip().lower()
    otp = data.get("otp")
    new_password = data.get("password")

    if not email or not otp or not new_password:
        return jsonify({
            "error": "missing_fields"
        }), 400

    row = PasswordResetOTP.query.filter_by(
        email=email,
        otp=otp
    ).first()

    if not row:
        return jsonify({
            "error": "invalid_otp"
        }), 400

    if row.expires_at < datetime.utcnow():
        db.session.delete(row)
        db.session.commit()

        return jsonify({
            "error": "otp_expired"
        }), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({
            "error": "user_not_found"
        }), 404

    user.set_password(new_password)

    db.session.delete(row)

    db.session.commit()

    return jsonify({
        "message": "Password reset successful"
    }), 200
class PasswordResetOTP(db.Model):
    __tablename__ = "password_reset_otp"

    id = db.Column(db.Integer, primary_key=True)

    email = db.Column(
        db.String(255),
        nullable=False
    )

    otp = db.Column(
        db.String(6),
        nullable=False
    )

    expires_at = db.Column(
        db.DateTime,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

class EnergyUsage(db.Model):
    __tablename__ = "energy_usage"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    company    = db.Column(db.String(200), nullable=False)
    date       = db.Column(db.Date, nullable=False)
    kwh        = db.Column(db.Float, nullable=False)
    notes      = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user       = db.relationship("User", backref=db.backref("energy_usage", lazy=True))

    def to_dict(self):
        return {
            "id": self.id, "user_id": self.user_id, "company": self.company,
            "date": self.date.isoformat(), "kwh": self.kwh,
            "co2_kg": calculate_emission_from_kwh(self.kwh),
            "notes": self.notes, "created_at": self.created_at.isoformat(),
        }


class Goal(db.Model):
    __tablename__ = "goal"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    month          = db.Column(db.String(7), nullable=False)   # "YYYY-MM"
    kwh_target     = db.Column(db.Float, nullable=True)
    co2_target_kg  = db.Column(db.Float, nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    user           = db.relationship("User", backref=db.backref("goals", lazy=True))

    def to_dict(self):
        return {
            "id": self.id, "month": self.month,
            "kwh_target": self.kwh_target,
            "co2_target_kg": self.co2_target_kg,
        }


# ── DB auto-init ──────────────────────────────────────────────────────────────
with app.app_context():
    try:
        from sqlalchemy import inspect as sa_inspect, text
        inspector = sa_inspect(db.engine)
        tables    = inspector.get_table_names()
        if not tables:
            db.create_all()
            logger.info("DB initialised: %s", sa_inspect(db.engine).get_table_names())
        else:
            # Add new columns to existing DBs gracefully
            db.create_all()
            logger.info("DB tables: %s", tables)
    except Exception as e:
        logger.exception("DB init error: %s", e)

# ── Predictor lazy-load ───────────────────────────────────────────────────────
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is not None:
        return _predictor_instance
    try:
        from models.predictor import Predictor
        _predictor_instance = Predictor()
        logger.info("Predictor ready: %s", _predictor_instance.model_source)
    except Exception as e:
        logger.warning("Predictor init failed: %s", e)
        _predictor_instance = None
    return _predictor_instance

def auto_train_model():
    try:
        from models.predictor import PROPHET_PATH
    except Exception:
        return
    if os.path.exists(PROPHET_PATH):
        return
    logger.info("No trained model found — running train_model.py...")
    try:
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")], check=True)
        logger.info("Auto-training complete.")
    except Exception as e:
        logger.warning("Auto-training failed: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_send_alert(user):
    """Fire an email if this month's total exceeds the user's threshold."""
    if not user.alert_email_enabled or not user.alert_threshold_kwh:
        return
    month_start = date.today().replace(day=1)
    total = db.session.query(db.func.sum(EnergyUsage.kwh)).filter(
        EnergyUsage.user_id == user.id,
        EnergyUsage.date >= month_start,
    ).scalar() or 0.0
    if total >= user.alert_threshold_kwh:
        try:
            import resend
            resend.api_key = os.environ.get("RESEND_API_KEY", "")
            resend.Emails.send({
                "from":    "EcoTrack <alerts@ecotrack.app>",
                "to":      [user.email],
                "subject": f"EcoTrack: monthly limit reached ({total:.1f} kWh)",
                "html":    f"""
                    <p>Hi {user.name or 'there'},</p>
                    <p>Your energy usage this month has reached
                    <strong>{total:.1f} kWh</strong>, exceeding your alert
                    threshold of <strong>{user.alert_threshold_kwh:.1f} kWh</strong>.</p>
                    <p>Log in to EcoTrack to review your usage.</p>
                """,
            })
            logger.info("Alert email sent to %s (%.1f kWh)", user.email, total)
        except Exception as e:
            logger.warning("Alert email failed: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
# Routes — core
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({"message": "EcoTrack AI backend", "status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()}), 200

# ─────────────────────────────────────────────────────────────────────────────
# Routes — auth
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def register():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password")
    name     = data.get("name")
    if not email or not password:
        return jsonify({"error": "email_and_password_required"}), 400
    if len(password) < 8:
        return jsonify({"error": "password_too_short", "min_length": 8}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email_already_registered"}), 400
    user = User(email=email, name=name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"token": create_jwt_for_user(user), "user": user.to_public()}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "email_and_password_required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not user.verify_password(password):
        return jsonify({"error": "invalid_credentials"}), 401
    return jsonify({"token": create_jwt_for_user(user), "user": user.to_public()}), 200

@app.route("/auth/me", methods=["GET"])
@jwt_required
def me():
    return jsonify({"user": g.current_user.to_public()}), 200

@app.route("/auth/send-reset-otp", methods=["POST"])
def send_reset_otp():
    data = request.get_json() or {}

    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({
            "error": "email_required"
        }), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({
            "error": "user_not_found"
        }), 404

    otp = str(random.randint(100000, 999999))

    PasswordResetOTP.query.filter_by(email=email).delete()

    reset = PasswordResetOTP(
        email=email,
        otp=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=10)
    )

    db.session.add(reset)
    db.session.commit()


    try:
        import resend

        resend.api_key = os.environ.get("RESEND_API_KEY")

        resend.Emails.send({
            "from": "EcoTrack <onboarding@resend.dev>",
            "to": [email],
            "subject": "EcoTrack Password Reset OTP",
            "html": f"""
                <h2>Password Reset OTP</h2>

                <p>Your OTP is:</p>

                <h1>{otp}</h1>

                <p>This OTP expires in 10 minutes.</p>
            """
        })

    except Exception as e:
        logger.exception("OTP send failed: %s", e)

        return jsonify({
            "error": "email_send_failed"
        }), 500

    return jsonify({
        "message": "OTP sent"
    }), 200

# ─────────────────────────────────────────────────────────────────────────────
# Routes — energy usage
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/save-energy-usage", methods=["POST"])
@jwt_required
def save_energy_usage():
    data    = request.get_json() or {}
    company = (data.get("company") or "").strip()
    kwh     = data.get("kwh")
    date_str= data.get("date")
    if not company or kwh is None or not date_str:
        return jsonify({"error": "company, kwh, and date are required"}), 400
    try:
        kwh = float(kwh)
    except ValueError:
        return jsonify({"error": "kwh must be a number"}), 400
    try:
        record_date = datetime.fromisoformat(date_str).date()
    except ValueError:
        return jsonify({"error": "date must be YYYY-MM-DD"}), 400

    record = EnergyUsage(
        user_id=g.current_user.id, company=company,
        date=record_date, kwh=kwh, notes=data.get("notes"),
    )
    db.session.add(record)
    db.session.commit()
    _maybe_send_alert(g.current_user)
    return jsonify({"record": record.to_dict()}), 201

@app.route("/calculate-emission", methods=["POST"])
@jwt_required
def calculate_emission():
    data = request.get_json() or {}
    try:
        kwh = float(data.get("kwh", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "kwh must be a number"}), 400
    return jsonify({"kwh": kwh, "co2_kg": calculate_emission_from_kwh(kwh)}), 200

@app.route("/history", methods=["GET"])
@jwt_required
def get_history():
    company   = request.args.get("company")
    from_date = request.args.get("from")
    to_date   = request.args.get("to")
    query     = EnergyUsage.query.filter_by(user_id=g.current_user.id)
    if company:
        query = query.filter_by(company=company)
    if from_date:
        try:
            query = query.filter(EnergyUsage.date >= datetime.fromisoformat(from_date).date())
        except Exception:
            return jsonify({"error": "invalid_from_date"}), 400
    if to_date:
        try:
            query = query.filter(EnergyUsage.date <= datetime.fromisoformat(to_date).date())
        except Exception:
            return jsonify({"error": "invalid_to_date"}), 400
    rows = query.order_by(EnergyUsage.date.asc()).all()
    return jsonify([r.to_dict() for r in rows]), 200

@app.route("/companies", methods=["GET"])
@jwt_required
def get_companies():
    names = (
        db.session.query(EnergyUsage.company)
        .filter_by(user_id=g.current_user.id)
        .distinct().order_by(EnergyUsage.company.asc()).all()
    )
    # Include per-company stats
    result = []
    for (name,) in names:
        rows = EnergyUsage.query.filter_by(user_id=g.current_user.id, company=name).all()
        total_kwh = sum(r.kwh for r in rows)
        total_co2 = calculate_emission_from_kwh(total_kwh)
        result.append({"name": name, "total_kwh": round(total_kwh, 4), "total_co2_kg": round(total_co2, 4)})
    return jsonify(result), 200

@app.route("/delete-energy-usage", methods=["POST"])
@jwt_required
def delete_energy_usage():
    data   = request.get_json() or {}
    rec_id = data.get("id")
    if not rec_id:
        return jsonify({"error": "id_required"}), 400
    rec = EnergyUsage.query.filter_by(id=rec_id, user_id=g.current_user.id).first()
    if not rec:
        return jsonify({"error": "not_found"}), 404
    db.session.delete(rec)
    db.session.commit()
    return jsonify({"deleted": rec_id}), 200

@app.route("/update-energy-usage", methods=["POST"])
@jwt_required
def update_energy_usage():
    data   = request.get_json() or {}
    rec_id = data.get("id")
    if not rec_id:
        return jsonify({"error": "id_required"}), 400
    rec = EnergyUsage.query.filter_by(id=rec_id, user_id=g.current_user.id).first()
    if not rec:
        return jsonify({"error": "not_found"}), 404
    rec.company = data.get("company", rec.company)
    if "date" in data:
        rec.date = datetime.fromisoformat(data["date"]).date()
    if "kwh" in data:
        rec.kwh = float(data["kwh"])
    rec.notes = data.get("notes", rec.notes)
    db.session.commit()
    return jsonify({"updated": rec.to_dict()}), 200

# ─────────────────────────────────────────────────────────────────────────────
# Routes — ML prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict-trend", methods=["POST"])
@jwt_required
def predict_trend():
    data    = request.get_json() or {}
    days    = int(data.get("days", 7))
    company = data.get("company")
    days    = max(1, min(days, 90))

    query = EnergyUsage.query.filter_by(user_id=g.current_user.id)
    if company:
        query = query.filter_by(company=company)
    history = query.order_by(EnergyUsage.date.asc()).all()

    history_df = None
    if history:
        try:
            history_df = pd.DataFrame([{"date": r.date, "kwh": r.kwh} for r in history])
            history_df["date"] = pd.to_datetime(history_df["date"])
        except Exception:
            history_df = None

    predictor = get_predictor()
    if predictor is None:
        return jsonify({"error": "predictor_unavailable"}), 503

    try:
        preds = predictor.predict_next_days(history_df, days=days)
        return jsonify({"predictions": preds, "model_type": predictor.model_source}), 200
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "prediction_failed", "details": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# Routes — sustainability goals  (Step 5)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/goals", methods=["GET"])
@jwt_required
def get_goals():
    month  = request.args.get("month", date.today().strftime("%Y-%m"))
    goal   = Goal.query.filter_by(user_id=g.current_user.id, month=month).first()
    if not goal:
        return jsonify({"goal": None}), 200
    # Calculate actual progress for the month
    month_start = datetime.strptime(month, "%Y-%m").date()
    month_end   = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    rows        = EnergyUsage.query.filter(
        EnergyUsage.user_id == g.current_user.id,
        EnergyUsage.date >= month_start,
        EnergyUsage.date <= month_end,
    ).all()
    actual_kwh    = sum(r.kwh for r in rows)
    actual_co2_kg = calculate_emission_from_kwh(actual_kwh)
    result = goal.to_dict()
    result["actual_kwh"]    = round(actual_kwh, 4)
    result["actual_co2_kg"] = round(actual_co2_kg, 4)
    result["kwh_pct"]       = round(actual_kwh / goal.kwh_target * 100, 1) if goal.kwh_target else None
    result["co2_pct"]       = round(actual_co2_kg / goal.co2_target_kg * 100, 1) if goal.co2_target_kg else None
    return jsonify({"goal": result}), 200

@app.route("/goals", methods=["POST"])
@jwt_required
def set_goal():
    data  = request.get_json() or {}
    month = data.get("month", date.today().strftime("%Y-%m"))
    goal  = Goal.query.filter_by(user_id=g.current_user.id, month=month).first()
    if not goal:
        goal = Goal(user_id=g.current_user.id, month=month)
        db.session.add(goal)
    if "kwh_target" in data:
        goal.kwh_target = float(data["kwh_target"]) if data["kwh_target"] else None
    if "co2_target_kg" in data:
        goal.co2_target_kg = float(data["co2_target_kg"]) if data["co2_target_kg"] else None
    db.session.commit()
    return jsonify({"goal": goal.to_dict()}), 200

# ─────────────────────────────────────────────────────────────────────────────
# Routes — export  (Step 7)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/export/csv", methods=["GET"])
@jwt_required
def export_csv():
    rows = EnergyUsage.query.filter_by(user_id=g.current_user.id) \
        .order_by(EnergyUsage.date.asc()).all()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["date","company","kwh","co2_kg","notes"])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "date": r.date.isoformat(), "company": r.company,
            "kwh": r.kwh, "co2_kg": round(calculate_emission_from_kwh(r.kwh), 4),
            "notes": r.notes or "",
        })
    resp = make_response(output.getvalue())
    resp.headers["Content-Type"]        = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=ecotrack_history.csv"
    return resp

@app.route("/export/pdf", methods=["GET"])
@jwt_required
def export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return jsonify({"error": "reportlab not installed"}), 500

    rows = EnergyUsage.query.filter_by(user_id=g.current_user.id) \
        .order_by(EnergyUsage.date.asc()).all()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("EcoTrack — Energy Usage Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    total_kwh = sum(r.kwh for r in rows)
    total_co2 = calculate_emission_from_kwh(total_kwh)
    story.append(Paragraph(f"Total: {total_kwh:.2f} kWh  ·  {total_co2:.2f} kg CO₂", styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = [["Date", "Company", "kWh", "CO₂ (kg)", "Notes"]]
    for r in rows:
        table_data.append([
            r.date.isoformat(), r.company, f"{r.kwh:.2f}",
            f"{calculate_emission_from_kwh(r.kwh):.2f}", r.notes or "",
        ])

    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1D9E75")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.HexColor("#B4B2A9")),
        ("PADDING",     (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    doc.build(story)

    buf.seek(0)
    resp = make_response(buf.read())
    resp.headers["Content-Type"]        = "application/pdf"
    resp.headers["Content-Disposition"] = "attachment; filename=ecotrack_report.pdf"
    return resp

# ─────────────────────────────────────────────────────────────────────────────
# Routes — AI eco tips  (Step 8)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/tips", methods=["GET"])
@jwt_required
def get_tips():

    import json

    user  = g.current_user
    today = date.today()

    # =========================
    # CACHE
    # =========================

    if (
        user.tips_cache and
        user.tips_cache_date == today
    ):

        try:

            return jsonify({
                "tips": json.loads(user.tips_cache),
                "cached": True
            }), 200

        except Exception:
            pass

    # =========================
    # OPENAI KEY
    # =========================

    OPENAI_API_KEY = os.environ.get(
        "OPENAI_API_KEY",
        ""
    )

    if not OPENAI_API_KEY:

        return jsonify({
            "error":
            "OPENAI_API_KEY not configured"
        }), 503

    # =========================
    # FETCH DATA
    # =========================

    since = today - timedelta(days=30)

    rows = EnergyUsage.query.filter(
        EnergyUsage.user_id == user.id,
        EnergyUsage.date >= since,
    ).all()

    # =========================
    # NO DATA
    # =========================

    if not rows:

        return jsonify({
            "tips": [

                "Start logging your energy usage regularly to unlock personalized sustainability insights.",

                "Track at least one week of energy usage data to improve AI recommendations.",

                "Compare multiple companies or locations to identify efficiency gaps."
            ],

            "cached": False

        }), 200

    # =========================
    # CALCULATIONS
    # =========================

    total_kwh = sum(r.kwh for r in rows)

    total_co2 = calculate_emission_from_kwh(
        total_kwh
    )

    companies = list({
        r.company for r in rows
    })

    peak_day = max(
        rows,
        key=lambda r: r.kwh
    )

    avg_daily_kwh = (
        total_kwh /
        max(len(set(r.date for r in rows)), 1)
    )

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are an expert sustainability and energy optimization advisor.

User's last 30 days energy usage:

- Total consumption: {total_kwh:.1f} kWh
- Estimated emissions: {total_co2:.1f} kg CO2
- Companies/sites: {', '.join(companies)}
- Daily average: {avg_daily_kwh:.1f} kWh
- Peak usage day: {peak_day.date} with {peak_day.kwh:.1f} kWh ({peak_day.company})

Provide exactly 3 practical, specific, actionable recommendations to reduce energy consumption and carbon emissions.

Return ONLY a valid JSON array of strings.

Example:
[
  "Tip 1",
  "Tip 2",
  "Tip 3"
]
"""

    try:

        client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {
                    "role": "system",

                    "content":
                    "You are a professional sustainability consultant."
                },

                {
                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0.7,

            max_tokens=300,
        )

        raw = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        tips = json.loads(raw)

        if not isinstance(tips, list):

            raise ValueError(
                "Invalid AI response format"
            )

        # =========================
        # CACHE
        # =========================

        user.tips_cache = json.dumps(tips)

        user.tips_cache_date = today

        db.session.commit()

        return jsonify({
            "tips": tips,
            "cached": False
        }), 200

    except Exception as e:

        logger.exception(
            "AI tips generation failed: %s",
            e
        )

        return jsonify({

            "tips": [

                "Reduce unnecessary lighting and equipment usage during non-operational hours.",

                "Monitor peak energy consumption periods to identify optimization opportunities.",

                "Replace inefficient appliances or systems with energy-efficient alternatives."
            ],

            "fallback": True,

            "error":
            str(e)

        }), 200
# ─────────────────────────────────────────────────────────────────────────────
# Routes — company benchmarking  (Step 9)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/benchmarks", methods=["GET"])
@jwt_required
def get_benchmarks():
    from_date = request.args.get("from")
    to_date   = request.args.get("to")
    query     = EnergyUsage.query.filter_by(user_id=g.current_user.id)
    if from_date:
        try:
            query = query.filter(EnergyUsage.date >= datetime.fromisoformat(from_date).date())
        except Exception:
            return jsonify({"error": "invalid_from_date"}), 400
    if to_date:
        try:
            query = query.filter(EnergyUsage.date <= datetime.fromisoformat(to_date).date())
        except Exception:
            return jsonify({"error": "invalid_to_date"}), 400

    rows = query.all()
    if not rows:
        return jsonify({"benchmarks": []}), 200

    # Aggregate per company
    by_company = {}
    for r in rows:
        if r.company not in by_company:
            by_company[r.company] = {"kwh": 0.0, "days": set(), "records": 0}
        by_company[r.company]["kwh"]    += r.kwh
        by_company[r.company]["days"].add(r.date)
        by_company[r.company]["records"] += 1

    total_kwh = sum(v["kwh"] for v in by_company.values())
    result    = []
    for company, data in sorted(by_company.items(), key=lambda x: -x[1]["kwh"]):
        kwh = data["kwh"]
        result.append({
            "company":   company,
            "total_kwh": round(kwh, 4),
            "co2_kg":    round(calculate_emission_from_kwh(kwh), 4),
            "days":      len(data["days"]),
            "records":   data["records"],
            "avg_daily_kwh": round(kwh / max(len(data["days"]), 1), 4),
            "pct_of_total":  round(kwh / total_kwh * 100, 1) if total_kwh > 0 else 0,
        })
    return jsonify({"benchmarks": result, "total_kwh": round(total_kwh, 4)}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Routes — user settings
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/user/settings", methods=["PATCH"])
@jwt_required
def patch_user_settings():
    data = request.get_json() or {}
    user = g.current_user
    if "alert_email_enabled" in data:
        user.alert_email_enabled = bool(data["alert_email_enabled"])
    if "alert_threshold_kwh" in data:
        val = data["alert_threshold_kwh"]
        user.alert_threshold_kwh = float(val) if val is not None else None
    if "name" in data:
        user.name = str(data["name"])[:200]
    db.session.commit()
    return jsonify({"user": user.to_public()}), 200

# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

@app.cli.command("init-db")
def init_db():
    with app.app_context():
        db.create_all()
        from sqlalchemy import inspect as sa_inspect
        logger.info("Tables: %s", sa_inspect(db.engine).get_table_names())
    logger.info("Database initialised.")

# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    auto_train_model()
    logger.info("Starting EcoTrack on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
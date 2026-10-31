"""DiaGnostix - Diabetes Prediction System (Flask server)."""
import os
import sqlite3
import sys
from datetime import datetime
from functools import wraps

import joblib
import pandas as pd
from flask import (Flask, flash, g, redirect, render_template, request,
                   session, url_for)
from sklearn.base import BaseEstimator, TransformerMixin
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_diabetes_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "diagnostix.db")


class IQRCapper(BaseEstimator, TransformerMixin):
    """Custom transformer used when the model was trained.

    Clips each column in ``cols`` to the IQR bounds learned during fit
    (stored in ``bounds_``). It must exist here so the pickle can load.
    """

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.bounds_ = {}
        for c in self.cols:
            q1, q3 = X[c].quantile(0.25), X[c].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            lo, hi = self.bounds_[c]
            X[c] = X[c].clip(lo, hi)
        return X


# The model was pickled from a script, so the class is looked up in __main__.
sys.modules["__main__"].IQRCapper = IQRCapper
model = joblib.load(MODEL_PATH)

# The 9 features in the exact order the pipeline was trained on.
FEATURES = [
    {"name": "Age", "label": "Age", "unit": "years", "min": 18, "max": 100, "step": 1,
     "placeholder": "e.g. 45", "help": "Your age in completed years."},
    {"name": "Sex", "label": "Sex", "unit": "", "type": "select",
     "options": [("1", "Male"), ("0", "Female")], "help": "Biological sex."},
    {"name": "BMI", "label": "Body Mass Index", "unit": "kg/m²", "min": 10, "max": 80, "step": 0.1,
     "placeholder": "e.g. 27.5", "help": "Weight (kg) ÷ height (m)². Use the calculator below."},
    {"name": "WaistCircumference", "label": "Waist Circumference", "unit": "cm", "min": 40, "max": 200,
     "step": 0.1, "placeholder": "e.g. 95", "help": "Measured at the navel level."},
    {"name": "SystolicBP", "label": "Systolic Blood Pressure", "unit": "mmHg", "min": 60, "max": 250,
     "step": 0.1, "placeholder": "e.g. 120", "help": "The upper number of a BP reading."},
    {"name": "DiastolicBP", "label": "Diastolic Blood Pressure", "unit": "mmHg", "min": 30, "max": 150,
     "step": 0.1, "placeholder": "e.g. 80", "help": "The lower number of a BP reading."},
    {"name": "HDL_Cholesterol", "label": "HDL Cholesterol", "unit": "mg/dL", "min": 10, "max": 200,
     "step": 0.1, "placeholder": "e.g. 50", "help": "\"Good\" cholesterol from a lipid panel."},
    {"name": "Total_Cholesterol", "label": "Total Cholesterol", "unit": "mg/dL", "min": 50, "max": 500,
     "step": 0.1, "placeholder": "e.g. 190", "help": "Total cholesterol from a lipid panel."},
    {"name": "SedentaryMinutes", "label": "Sedentary Time", "unit": "min/day", "min": 0, "max": 1440,
     "step": 1, "placeholder": "e.g. 360", "help": "Minutes per day spent sitting (excluding sleep)."},
]
FEATURE_NAMES = [f["name"] for f in FEATURES]

app = Flask(__name__)
app.secret_key = os.environ.get("DIAGNOSTIX_SECRET", "change-this-secret-key-in-production")


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                patient_name TEXT,
                age REAL, sex INTEGER, bmi REAL, waist REAL,
                systolic REAL, diastolic REAL, hdl REAL, total_chol REAL,
                sedentary REAL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                created_at TEXT NOT NULL
            );
        """)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------- helpers
def risk_level(prob):
    if prob < 0.20:
        return "Low", "low"
    if prob < 0.40:
        return "Moderate", "moderate"
    return "High", "high"


def build_insights(values):
    """Simple, rule-based notes on individual readings (informational only)."""
    notes = []
    bmi = values["BMI"]
    if bmi >= 30:
        notes.append(("high", "BMI", f"{bmi:.1f} is in the obese range (≥ 30)."))
    elif bmi >= 25:
        notes.append(("moderate", "BMI", f"{bmi:.1f} is in the overweight range (25–29.9)."))
    elif bmi < 18.5:
        notes.append(("moderate", "BMI", f"{bmi:.1f} is in the underweight range (< 18.5)."))
    else:
        notes.append(("low", "BMI", f"{bmi:.1f} is in the healthy range."))

    waist_limit = 102 if values["Sex"] == 1 else 88
    if values["WaistCircumference"] > waist_limit:
        notes.append(("high", "Waist", f"Above {waist_limit} cm — indicates abdominal obesity."))
    else:
        notes.append(("low", "Waist", "Within the recommended range."))

    sbp, dbp = values["SystolicBP"], values["DiastolicBP"]
    if sbp >= 140 or dbp >= 90:
        notes.append(("high", "Blood pressure", f"{sbp:.0f}/{dbp:.0f} mmHg — stage 2 hypertension range."))
    elif sbp >= 130 or dbp >= 80:
        notes.append(("moderate", "Blood pressure", f"{sbp:.0f}/{dbp:.0f} mmHg — stage 1 hypertension range."))
    elif sbp >= 120:
        notes.append(("moderate", "Blood pressure", f"{sbp:.0f}/{dbp:.0f} mmHg — elevated."))
    else:
        notes.append(("low", "Blood pressure", f"{sbp:.0f}/{dbp:.0f} mmHg — normal."))

    hdl_limit = 40 if values["Sex"] == 1 else 50
    if values["HDL_Cholesterol"] < hdl_limit:
        notes.append(("high", "HDL", f"Below {hdl_limit} mg/dL — low protective cholesterol."))
    else:
        notes.append(("low", "HDL", "At a healthy level."))

    tc = values["Total_Cholesterol"]
    if tc >= 240:
        notes.append(("high", "Total cholesterol", f"{tc:.0f} mg/dL — high."))
    elif tc >= 200:
        notes.append(("moderate", "Total cholesterol", f"{tc:.0f} mg/dL — borderline high."))
    else:
        notes.append(("low", "Total cholesterol", f"{tc:.0f} mg/dL — desirable."))

    sed = values["SedentaryMinutes"]
    if sed >= 480:
        notes.append(("high", "Sedentary time", f"{sed / 60:.1f} h/day — try to break up long sitting periods."))
    elif sed >= 360:
        notes.append(("moderate", "Sedentary time", f"{sed / 60:.1f} h/day — on the higher side."))
    else:
        notes.append(("low", "Sedentary time", f"{sed / 60:.1f} h/day."))
    return notes


# ---------------------------------------------------------------- auth routes
@app.route("/")
def index():
    return redirect(url_for("dashboard" if "user_id" in session else "login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip().lower()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        error = None
        if not all([full_name, username, email, password]):
            error = "All fields are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif "@" not in email:
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."

        if error is None:
            db = get_db()
            exists = db.execute("SELECT 1 FROM users WHERE username = ? OR email = ?",
                                (username, email)).fetchone()
            if exists:
                error = "That username or email is already registered."
            else:
                db.execute(
                    "INSERT INTO users (full_name, username, email, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (full_name, username, email, generate_password_hash(password),
                     datetime.now().isoformat(timespec="seconds")))
                db.commit()
                flash("Account created successfully. Please log in.", "success")
                return redirect(url_for("login"))
        flash(error, "danger")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        identifier = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE username = ? OR email = ?",
                                (identifier, identifier)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["full_name"] = user["full_name"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['full_name'].split()[0]}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---------------------------------------------------------------- app routes
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    stats = db.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(prediction), 0) AS positive, "
        "AVG(probability) AS avg_prob FROM predictions WHERE user_id = ?",
        (session["user_id"],)).fetchone()
    recent = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 5",
        (session["user_id"],)).fetchall()
    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return render_template("predict.html", features=FEATURES, form={})

    values, errors = {}, []
    for f in FEATURES:
        raw = request.form.get(f["name"], "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{f['label']} must be a number.")
            continue
        if f.get("type") == "select":
            if raw not in {o[0] for o in f["options"]}:
                errors.append(f"Please select a valid {f['label'].lower()}.")
                continue
            val = int(val)
        elif not f["min"] <= val <= f["max"]:
            errors.append(f"{f['label']} must be between {f['min']} and {f['max']} {f['unit']}.")
            continue
        values[f["name"]] = val

    if errors:
        for e in errors:
            flash(e, "danger")
        return render_template("predict.html", features=FEATURES, form=request.form)

    X = pd.DataFrame([values], columns=FEATURE_NAMES)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    patient_name = request.form.get("patient_name", "").strip() or session["full_name"]

    db = get_db()
    cur = db.execute(
        "INSERT INTO predictions (user_id, patient_name, age, sex, bmi, waist, systolic, diastolic, "
        "hdl, total_chol, sedentary, prediction, probability, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session["user_id"], patient_name, *[values[n] for n in FEATURE_NAMES],
         prediction, probability, datetime.now().isoformat(timespec="seconds")))
    db.commit()
    return redirect(url_for("result", pred_id=cur.lastrowid))


@app.route("/result/<int:pred_id>")
@login_required
def result(pred_id):
    row = get_db().execute("SELECT * FROM predictions WHERE id = ? AND user_id = ?",
                           (pred_id, session["user_id"])).fetchone()
    if row is None:
        flash("Prediction not found.", "danger")
        return redirect(url_for("history"))
    values = dict(zip(FEATURE_NAMES, [row["age"], row["sex"], row["bmi"], row["waist"], row["systolic"],
                                      row["diastolic"], row["hdl"], row["total_chol"], row["sedentary"]]))
    level, level_cls = risk_level(row["probability"])
    return render_template("result.html", row=row, values=values, features=FEATURES,
                           level=level, level_cls=level_cls, insights=build_insights(values))


@app.route("/history")
@login_required
def history():
    rows = get_db().execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
                            (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows, risk_level=risk_level)


@app.route("/history/<int:pred_id>/delete", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (pred_id, session["user_id"]))
    db.commit()
    flash("Record deleted.", "info")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html", features=FEATURES)


init_db()

if __name__ == "__main__":
    app.run(debug=True)

# DiaGnostix — Diabetes Prediction System

DiaGnostix is a Flask web application that estimates a person's risk of diabetes from nine routine health measurements. It uses a Support Vector Machine (SVM) trained on the **NHANES 2021–2023** dataset and wraps it in a multi-user web interface with login, prediction history and result reports.

## Features

- **User accounts** — registration and login, with passwords stored as salted hashes (Werkzeug) in SQLite.
- **Dashboard** — total assessments, number predicted diabetic/non-diabetic, average risk and recent results.
- **Prediction form** — nine inputs grouped by category, with range validation and a built-in BMI calculator.
- **Result report** — probability gauge, model verdict, a Low / Moderate / High risk level, and rule-based notes on each reading using common clinical reference ranges. Print-friendly.
- **History** — every prediction saved per user, searchable by name, with delete.
- **Responsive UI** — plain HTML/CSS (Jinja templates), works on desktop and mobile.

## Model

The saved model (`svm_diabetes_model.pkl`) is a scikit-learn `Pipeline`:

1. **IQRCapper** — custom transformer that clips outliers to the 1.5 × IQR bounds learned during training
2. **SimpleImputer** — median imputation for missing values
3. **StandardScaler** — feature standardisation
4. **SVC** — RBF kernel, `C=1.0`, `probability=True`

The target is the `diabetes` column (0 = no diabetes, 1 = diabetes). On the full dataset the pipeline scores about **84 % accuracy**.

### Input features

| Feature | Description | Unit | Dataset column |
|---|---|---|---|
| Age | Age in years | years | `age_years` |
| Sex | 1 = Male, 0 = Female | — | `sex` |
| BMI | Body mass index | kg/m² | `bmi_kg_m2` |
| WaistCircumference | Waist circumference | cm | `waist_circumference_cm` |
| SystolicBP | Average systolic blood pressure | mmHg | `avg_systolic_bp_mmhg` |
| DiastolicBP | Average diastolic blood pressure | mmHg | `avg_diastolic_bp_mmhg` |
| HDL_Cholesterol | HDL cholesterol | mg/dL | `hdl_cholesterol_mg_dl` |
| Total_Cholesterol | Total cholesterol | mg/dL | `total_cholesterol_mg_dl` |
| SedentaryMinutes | Time spent sitting per day | min/day | `sedentary_minutes_day` |

### Risk levels

Alongside the model's class prediction, the result page shows a risk level based on the predicted probability:

| Level | Probability |
|---|---|
| Low | < 20 % |
| Moderate | 20 – 40 % |
| High | ≥ 40 % |

The SVM is conservative: it labels only about 3 % of the dataset as diabetic, while about 17.5 % actually are. The risk level is therefore useful on its own — in the dataset, 54 % of people scoring 20 % or higher have diabetes, compared with 11 % of those below 20 %.

## Project structure

```
Diabetes/
├── app.py                          # Flask server: auth, prediction, history, IQRCapper class
├── svm_diabetes_model.pkl          # Trained SVM pipeline
├── nhanes_diabetes_2021_2023.csv   # Training dataset (NHANES 2021–2023)
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    ├── base.html                   # Layout, navigation, flash messages
    ├── auth_layout.html            # Shared layout for login/register
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── predict.html
    ├── result.html
    ├── history.html
    └── about.html
```

## Getting started

### Requirements

- Python 3.10+
- scikit-learn **1.6.1** (the version the model was saved with)

### Installation

```bash
git clone <your-repo-url>
cd Diabetes
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**, create an account and log in. The SQLite database `diagnostix.db` is created automatically on first run.

### Configuration

| Environment variable | Purpose |
|---|---|
| `DIAGNOSTIX_SECRET` | Flask session secret key. Set this to a long random value before deploying. |

For anything beyond local use, also set `debug=False` in `app.py` and serve the app with a production WSGI server such as Waitress or Gunicorn.

## Sample input

These values produce a **diabetic** prediction (≈ 96 % probability):

| Age | Sex | BMI | Waist | Systolic | Diastolic | HDL | Total chol. | Sedentary |
|---|---|---|---|---|---|---|---|---|
| 68 | Male | 36 | 122 | 150 | 65 | 32 | 130 | 720 |

## Routes

| Route | Method | Description |
|---|---|---|
| `/register` | GET, POST | Create an account |
| `/login` | GET, POST | Log in with username or email |
| `/logout` | GET | Log out |
| `/dashboard` | GET | Summary and recent predictions |
| `/predict` | GET, POST | Prediction form / run a prediction |
| `/result/<id>` | GET | Detailed result for one prediction |
| `/history` | GET | All of the user's predictions |
| `/history/<id>/delete` | POST | Delete a prediction |
| `/about` | GET | About the model |

## Tech stack

Flask · scikit-learn · pandas · NumPy · SQLite · HTML / CSS / Jinja2

## Dataset

National Health and Nutrition Examination Survey (NHANES), 2021–2023 cycle, published by the U.S. Centers for Disease Control and Prevention (CDC) / National Center for Health Statistics.

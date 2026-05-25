📄 EcoTrack AI – Backend

A robust backend built using Flask, SQLite ORM (SQLAlchemy), scikit-learn, and a complete ML pipeline for forecasting energy consumption.
It exposes REST APIs for authentication, data upload, analytics, and predictions.

🚀 Live Backend URL

🔗 https://your-backend-render-url.onrender.com

🌟 Key Features
🔐 Authentication System

JWT-based login & registration

Secure password hashing (bcrypt + PBKDF2)

Auth-required dashboard endpoints

📊 Energy Data Storage

Stores daily kWh records

Company-wise consumption

Notes, filters, and time-range analytics

🤖 ML Prediction Engine

Train/Test split (automated)

Linear Regression model

Predicts future usage

Supports multiple companies

📁 CSV Upload Endpoint

Upload CSV from frontend

Validates rows

Inserts into DB

Retrains ML model on the fly

📈 Advanced Analytics

Trend detection

Peak days

Cost estimation

Carbon emission estimation

🏗 Tech Stack
Area	Technology
Framework	Flask 2.3
Database	SQLite (local) / PostgreSQL optional
ORM	SQLAlchemy
ML Engine	scikit-learn
Data Handling	pandas
Auth	JWT
Deployment	Render
🔌 Environment Variables

Create a .env file in backend root:

```
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
DATABASE_URL=sqlite:///ecotrack.db
```

When deploying on Render:
```
SECRET_KEY=****
JWT_SECRET=****
PYTHON_VERSION=3.11.6
```
📂 Project Structure
```
EcoTrack-AI-BackEnd/
│
├── app.py              # Main API + Routes
├── models/             # SQLAlchemy ORM models
├── ml/
│   ├── model.pkl       # Saved ML model
│   ├── train.py        # Training logic
│
├── utils/
│   ├── validators.py
│   ├── analytics.py
│
├── energy.db           # SQLite database
├── requirements.txt
└── README.md
```
🧪 Running Locally
Install dependencies:
```
pip install -r requirements.txt
```
Run Flask server:
```
python app.py
```

📌 Important API Endpoints
🧑‍💻 Auth
Method	Endpoint	Description
```POST	/auth/register	Create user
POST	/auth/login	Login user
```
⚡ Usage Data
Method	Endpoint	Description
```
GET	/history	Fetch filtered usage
POST	/upload	Upload CSV
GET	/analytics	Cost, emissions, trends
GET	/predict	AI forecasting
```
🔍 Health Check
GET /health

🛠 Deployment (Render)
1️⃣ Add Build Command

Render automatically detects Python backend.

2️⃣ Add Start Command
```
gunicorn app:app
```
3️⃣ Add environment variables
```
SECRET_KEY
JWT_SECRET
PYTHON_VERSION=3.11.6
```
4️⃣ Deploy 🚀

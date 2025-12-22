# Healthcare Analytics Platform

A production-grade healthcare analytics platform featuring **real-time anomaly detection**, **multi-model ML predictions**, and **explainable AI** for hospital operations optimization.

Built with FastAPI, Next.js, and modern ML techniques including SHAP explainability and multi-dimensional anomaly detection.

---

## Architecture

```
healthcare-analytics/
├── backend/                    # FastAPI REST API
│   ├── app/
│   │   ├── models/            # Pydantic schemas
│   │   ├── routers/           # API endpoints
│   │   └── services/          # Business logic
│   │       ├── anomaly_detector.py    # Multi-dimensional detection
│   │       ├── baseline_calculator.py # Seasonal baselines
│   │       └── predictor.py           # ML predictions + SHAP
│   └── requirements.txt
├── frontend/                   # Next.js 15 Dashboard
│   └── src/
│       ├── app/               # App Router pages
│       ├── components/        # Tremor UI components
│       └── lib/               # API client & types
├── models/                     # Trained ML models
├── notebooks/                  # Jupyter analysis
└── streamlit_app.py           # Legacy Streamlit dashboard
```

---

## Key Features

### 1. Real-Time Anomaly Detection
- **Seasonal Baselines**: Hour and day-of-week adjusted thresholds
- **Z-Score Alerting**: Configurable thresholds with severity levels
- **Isolation Forest**: Multi-dimensional anomaly detection across 5 metrics
- **Auto-Refresh**: 15-30 second polling intervals

### 2. Multi-Model ML Pipeline
- **8 Models Compared**: XGBoost, LightGBM, Random Forest, Gradient Boosting, Ridge, Lasso, ElasticNet, SVR
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Performance Metrics**: RMSE, MAE, R², with confidence intervals

### 3. Explainable AI (XAI)
- **SHAP Integration**: Feature importance and contribution analysis
- **Per-Prediction Explanations**: Individual prediction breakdowns
- **Base Value Reference**: Average prediction baseline for context

### 4. Production Dashboard
- **Next.js 15**: App Router with server components
- **Tremor UI**: Modern data visualization components
- **Dark Mode**: Professional healthcare-appropriate theme
- **Responsive Design**: Mobile and desktop optimized

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 15, React 19, Tremor, Tailwind CSS |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **ML/AI** | Scikit-learn, XGBoost, LightGBM, SHAP |
| **Data** | Pandas, NumPy, MIMIC-III |
| **Deployment** | Vercel (frontend), Railway (backend) |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install --legacy-peer-deps
cp .env.example .env.local
# Edit .env.local with your API URL
npm run dev
```

Open http://localhost:3000 for the dashboard.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/metrics/current` | Live hospital metrics |
| `GET` | `/api/metrics/history` | 24-hour metric history |
| `GET` | `/api/anomalies/active` | Current anomaly alerts |
| `GET` | `/api/anomalies/history` | Historical anomalies |
| `POST` | `/api/predict/wait-time` | ED wait time prediction |
| `POST` | `/api/predict/explain` | SHAP explanation |
| `GET` | `/api/predict/model-info` | Model metadata |
| `GET` | `/health` | Service health check |

---

## Deployment

### Frontend (Vercel)

1. Connect your GitHub repo to Vercel
2. Set environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
   ```
3. Deploy automatically on push

### Backend (Railway)

1. Connect your GitHub repo to Railway
2. Railway auto-detects the `Procfile`
3. Set any required environment variables
4. Deploy automatically on push

---

## Machine Learning Details

### Anomaly Detection Algorithm

```python
# Multi-dimensional scoring
1. Calculate Z-scores for each metric vs seasonal baseline
2. Flag univariate anomalies where |Z| > 2.5
3. Run Isolation Forest on normalized metric vector
4. Combine scores: multi_dim_score = isolation_score * 10
5. System anomalous if score > 5.0 or any critical alerts
```

### Prediction Model

- **Target**: Emergency Department wait time (minutes)
- **Features**: admission_type, hour, admission_location, ethnicity
- **Best Model**: XGBoost with hyperparameter tuning
- **Confidence Intervals**: 95% CI using prediction distribution

---

## Project Structure

```
Key Files:
├── backend/app/main.py                    # FastAPI app entry
├── backend/app/services/anomaly_detector.py  # Core detection logic
├── backend/app/services/predictor.py      # ML predictions
├── frontend/src/app/page.tsx              # Dashboard home
├── frontend/src/app/alerts/page.tsx       # Anomaly alerts view
├── frontend/src/app/predict/page.tsx      # Prediction interface
├── notebooks/model_comparison.ipynb       # ML experimentation
└── models/xgboost_model.pkl              # Trained model artifact
```

---

## Data Source

This project uses the **MIMIC-III Clinical Database**, a freely accessible critical care database containing de-identified health data from ICU patients.

> PhysioNet: https://physionet.org/content/mimiciii/

**Note**: This project is for educational and research purposes only.

---

## Screenshots

### Live Metrics Dashboard
Real-time monitoring with anomaly scoring and metric cards

### Anomaly Alerts
Detailed alert table with severity, Z-scores, and context

### ED Wait Time Prediction
Interactive prediction form with SHAP explanations

---

## Development

### Running Tests
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Code Quality
```bash
# Backend
ruff check .
black .

# Frontend
npm run lint
```

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with modern ML and web technologies for healthcare operations optimization.

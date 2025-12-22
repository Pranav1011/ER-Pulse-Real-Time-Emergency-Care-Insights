# Healthcare Analytics Platform Enhancement

## Design Document

**Date**: 2025-01-22
**Status**: Approved
**Goal**: Transform existing analytics project into a production-ready platform with multi-dimensional anomaly detection and modern web dashboard.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    VERCEL (Frontend)                    │
│  Next.js + Tremor Dashboard                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
│  │ Real-time   │ │  Anomaly    │ │   Prediction    │   │
│  │ Metrics     │ │  Alerts     │ │   Explorer      │   │
│  └─────────────┘ └─────────────┘ └─────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │ REST API calls
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  RAILWAY (Backend)                       │
│  FastAPI + ML Models                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐  │
│  │ Anomaly      │ │ Prediction   │ │ Metrics        │  │
│  │ Detection    │ │ Service      │ │ Aggregation    │  │
│  │ Engine       │ │ (ED Wait)    │ │ Service        │  │
│  └──────────────┘ └──────────────┘ └────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Seasonal Baseline Calculator + Rolling Window     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Anomaly Detection Engine

### Metrics Tracked

| Metric | Description | Why it matters |
|--------|-------------|----------------|
| ED Wait Time | Average wait in rolling windows | Core patient experience |
| Admission Rate | Admissions per hour | Demand indicator |
| Transfer Delays | Time between transfer events | Bottleneck signal |
| Department Load | Patients per ward | Capacity pressure |
| Length of Stay | Average LOS by department | Flow efficiency |

### Detection Approach

1. **Seasonal Baselines**: For each (hour, day-of-week) pair, calculate historical mean and standard deviation.

2. **Rolling Window**: Last 48 hours of data, calculate short-term trends and detect sudden shifts.

3. **Anomaly Scoring**: For each metric:
   - Z-score against seasonal baseline
   - Z-score against rolling window
   - Combined anomaly score = weighted average

4. **Multi-dimensional Alert**: Flag when:
   - Any single metric exceeds threshold (z > 2.5), OR
   - Multiple metrics are elevated together (sum of z-scores > threshold)

5. **Algorithm**: Isolation Forest trained on multi-dimensional feature space to catch unusual combinations.

---

## Frontend Dashboard

### Pages

#### 1. Live Metrics Dashboard (`/`)
- KPI Cards: Current ED wait, admission rate, department loads
- Sparklines: 24-hour trend for each metric
- Status indicators: Green/Yellow/Red based on anomaly scores
- Time selector: View last 6h, 24h, 7d

#### 2. Anomaly Alerts (`/alerts`)
- Alert timeline: Chronological list of detected anomalies
- Severity badges: Critical / Warning / Info
- Anomaly details: Which metrics triggered, how far from baseline
- Context panel: Shows seasonal expectation vs actual value

#### 3. Prediction Explorer (`/predict`)
- Interactive form: Input admission type, hour, location, ethnicity
- Prediction result: ED wait time with confidence interval
- SHAP explanation: Visual breakdown of feature contributions
- Historical comparison: "This is higher/lower than typical for these inputs"

### Design System
- Dark mode by default
- Tremor components throughout
- Responsive for desktop/tablet

---

## API Endpoints

```
/api
├── /metrics
│   ├── GET /current          # Latest values for all metrics
│   ├── GET /history          # Historical data with time range params
│   └── GET /baselines        # Seasonal baselines for comparison
│
├── /anomalies
│   ├── GET /active           # Currently active anomalies
│   ├── GET /history          # Past anomalies with filters
│   └── GET /score            # Real-time anomaly scores per metric
│
├── /predict
│   ├── POST /ed-wait         # Predict ED wait time
│   └── POST /explain         # SHAP explanation for a prediction
│
└── /health                   # Health check for Railway
```

---

## Project Structure

```
Healthcare-Analytics/
├── frontend/                    # Next.js app (Vercel)
│   ├── app/
│   │   ├── page.tsx            # Live metrics dashboard
│   │   ├── alerts/page.tsx     # Anomaly alerts
│   │   └── predict/page.tsx    # Prediction explorer
│   ├── components/
│   │   ├── MetricCard.tsx
│   │   ├── AnomalyTimeline.tsx
│   │   ├── ShapExplainer.tsx
│   │   └── ...
│   ├── lib/
│   │   └── api.ts              # API client
│   └── package.json
│
├── backend/                     # FastAPI app (Railway)
│   ├── app/
│   │   ├── main.py             # FastAPI entry
│   │   ├── routers/
│   │   │   ├── metrics.py
│   │   │   ├── anomalies.py
│   │   │   └── predict.py
│   │   ├── services/
│   │   │   ├── anomaly_detector.py
│   │   │   ├── baseline_calculator.py
│   │   │   └── predictor.py
│   │   └── models/             # Pydantic schemas
│   ├── requirements.txt
│   └── railway.json
│
├── src/                         # Existing ML code (reused)
├── data/                        # Existing data
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14, Tremor, Tailwind CSS |
| Backend | FastAPI, Pydantic |
| ML | scikit-learn (Isolation Forest), SHAP |
| Hosting | Vercel (frontend), Railway (backend) |

---

## Out of Scope

- User authentication
- Database (using static processed data)
- Real-time WebSockets (polling is sufficient)
- Email/SMS alerts
- Admin panel
- Model retraining UI
- Multiple hospital support

---

## Success Criteria

1. Dashboard loads and displays current metrics
2. Anomaly detection correctly identifies unusual patterns
3. Predictions return with SHAP explanations
4. Both frontend and backend deploy successfully
5. API documentation auto-generated and accessible

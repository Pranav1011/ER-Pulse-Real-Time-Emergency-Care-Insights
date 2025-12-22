from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import metrics, anomalies, predict

app = FastAPI(
    title="Healthcare Analytics API",
    description="Multi-dimensional anomaly detection and ED wait time prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metrics.router)
app.include_router(anomalies.router)
app.include_router(predict.router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "healthcare-analytics-api"}

@app.get("/")
def root():
    return {"message": "Healthcare Analytics API", "docs": "/docs"}

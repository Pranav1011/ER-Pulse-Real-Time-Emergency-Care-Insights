import type {
  MetricsResponse,
  AnomaliesResponse,
  AnomalyScoreResponse,
  PredictionRequest,
  PredictionResponse,
  ExplanationResponse,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export const api = {
  // Metrics
  getCurrentMetrics: () => fetchAPI<MetricsResponse>('/api/metrics/current'),

  getMetricsHistory: (hours: number = 24) =>
    fetchAPI<{ hours_requested: number; data: unknown[] }>(`/api/metrics/history?hours=${hours}`),

  getBaselines: () => fetchAPI<{ timestamp: string; baselines: Record<string, unknown[]> }>('/api/metrics/baselines'),

  // Anomalies
  getActiveAnomalies: () => fetchAPI<AnomaliesResponse>('/api/anomalies/active'),

  getAnomalyScores: () => fetchAPI<AnomalyScoreResponse>('/api/anomalies/score'),

  getAnomalyHistory: (hours: number = 24) =>
    fetchAPI<{ hours_requested: number; anomalies: unknown[] }>(`/api/anomalies/history?hours=${hours}`),

  // Predictions
  predictWaitTime: (data: PredictionRequest) =>
    fetchAPI<PredictionResponse>('/api/predict/ed-wait', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  explainPrediction: (data: PredictionRequest) =>
    fetchAPI<ExplanationResponse>('/api/predict/explain', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Health
  healthCheck: () => fetchAPI<{ status: string }>('/health'),
};

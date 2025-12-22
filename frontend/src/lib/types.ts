export interface MetricValue {
  current: number;
  baseline: number;
  z_score: number;
  severity: 'info' | 'warning' | 'critical';
  context: string;
}

export interface MetricsResponse {
  timestamp: string;
  metrics: {
    ed_wait_time: MetricValue;
    admission_rate: MetricValue;
    transfer_delay: MetricValue;
    department_load: MetricValue;
    length_of_stay: MetricValue;
  };
}

export interface AnomalyAlert {
  id: string;
  timestamp: string;
  metric: string;
  current: number;
  baseline: number;
  z_score: number;
  severity: 'info' | 'warning' | 'critical';
  context: string;
}

export interface AnomaliesResponse {
  timestamp: string;
  active_anomalies: AnomalyAlert[];
  multi_dimensional_score: number;
  is_system_anomalous: boolean;
}

export interface AnomalyScoreResponse {
  timestamp: string;
  scores: Record<string, number>;
  combined_score: number;
  is_anomalous: boolean;
}

export interface PredictionRequest {
  admission_type: string;
  hour: number;
  admission_location: string;
  ethnicity: string;
}

export interface PredictionResponse {
  timestamp: string;
  predicted_wait_time: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  unit: string;
}

export interface FeatureContribution {
  feature: string;
  shap_value: number;
  direction: string;
}

export interface ExplanationResponse {
  timestamp: string;
  prediction: number;
  base_value: number | null;
  top_contributors: FeatureContribution[];
}

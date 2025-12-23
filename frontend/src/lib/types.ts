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

// Multi-model types
export interface PatientInput {
  admission_type: string;
  admission_location: string;
  ethnicity: string;
  hour: number;
  day_of_week: number;
  age: number;
  prev_admissions: number;
  num_diagnoses: number;
}

export interface RegressionPrediction {
  value: number;
  unit: string;
  confidence_interval: [number, number];
}

export interface ClassificationPrediction {
  value: number;
  class: number;
  risk_level: 'Low' | 'Medium' | 'High';
  unit: string;
}

export interface MultiPredictionResponse {
  status: string;
  predictions: {
    ed_wait_time?: RegressionPrediction;
    length_of_stay?: RegressionPrediction;
    mortality_risk?: ClassificationPrediction;
  };
}

export interface ModelExplanation {
  target: string;
  prediction: number;
  base_value: number;
  contributors: {
    feature: string;
    impact: number;
    direction: string;
  }[];
}

export interface ForecastPoint {
  hours_ahead: number;
  predicted_load: number;
  hour: number;
  is_weekend: boolean;
}

export interface ForecastResponse {
  status: string;
  forecasts: ForecastPoint[];
}

export interface ModelMetric {
  mae?: number;
  r2?: number;
  accuracy?: number;
  auc?: number;
  f1?: number;
  type: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelMetricsResponse {
  status: string;
  metrics: Record<string, ModelMetric>;
  feature_importance: Record<string, FeatureImportance[]>;
  models_loaded: string[];
}

export interface ModelInfoResponse {
  models: Record<string, {
    description: string;
    type: string;
    target_unit: string;
  }>;
  features: string[];
  cat_features: string[];
  num_features: string[];
}

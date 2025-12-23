'use client';

import { useState, useEffect } from 'react';
import { Card, Title, Text, Grid, Badge, BarList, ProgressBar, Metric } from '@tremor/react';

interface ModelMetric {
  mae?: number;
  r2?: number;
  accuracy?: number;
  auc?: number;
  f1?: number;
  type: string;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

// Mock data for demo (replace with API call when backend is running)
const mockMetrics: Record<string, ModelMetric> = {
  ed_wait_time: { mae: 30.42, r2: 0.5728, type: 'regression' },
  length_of_stay: { mae: 1.71, r2: 0.3283, type: 'regression' },
  mortality_risk: { accuracy: 0.825, auc: 0.825, f1: 0.4, type: 'classification' },
  hospital_load: { mae: 6.02, r2: 0.743, type: 'timeseries' },
};

const mockImportance: Record<string, FeatureImportance[]> = {
  ed_wait_time: [
    { feature: 'hour', importance: 0.28 },
    { feature: 'admission_type_EMERGENCY', importance: 0.22 },
    { feature: 'admission_location_EMERGENCY ROOM', importance: 0.18 },
    { feature: 'age', importance: 0.12 },
    { feature: 'num_diagnoses', importance: 0.08 },
  ],
  mortality_risk: [
    { feature: 'age', importance: 0.32 },
    { feature: 'num_diagnoses', importance: 0.24 },
    { feature: 'prev_admissions', importance: 0.18 },
    { feature: 'admission_type_EMERGENCY', importance: 0.14 },
    { feature: 'hour', importance: 0.05 },
  ],
};

const modelDescriptions: Record<string, { name: string; description: string; icon: string }> = {
  ed_wait_time: {
    name: 'ED Wait Time',
    description: 'Predicts emergency department wait time in minutes',
    icon: '⏱️',
  },
  length_of_stay: {
    name: 'Length of Stay',
    description: 'Predicts hospital stay duration in days',
    icon: '🏥',
  },
  mortality_risk: {
    name: 'Mortality Risk',
    description: 'Classifies in-hospital mortality probability',
    icon: '⚕️',
  },
  hospital_load: {
    name: 'Hospital Load',
    description: 'Forecasts hospital occupancy percentage',
    icon: '📈',
  },
};

function MetricBadge({ value, type }: { value: number; type: 'r2' | 'accuracy' | 'auc' }) {
  let color: 'green' | 'yellow' | 'red' = 'red';
  if (type === 'r2') {
    color = value > 0.5 ? 'green' : value > 0.3 ? 'yellow' : 'red';
  } else {
    color = value > 0.8 ? 'green' : value > 0.6 ? 'yellow' : 'red';
  }
  return <Badge color={color}>{(value * 100).toFixed(1)}%</Badge>;
}

export default function ModelsPage() {
  const [metrics] = useState(mockMetrics);
  const [importance] = useState(mockImportance);

  return (
    <div className="space-y-6">
      <div>
        <Title>Model Performance Dashboard</Title>
        <Text>XGBoost models trained on 3,000 synthetic patient records</Text>
      </div>

      {/* Model Cards */}
      <Grid numItemsMd={2} numItemsLg={4} className="gap-4">
        {Object.entries(metrics).map(([key, metric]) => {
          const info = modelDescriptions[key];
          return (
            <Card key={key} decoration="top" decorationColor={metric.r2 && metric.r2 > 0.5 ? 'green' : metric.accuracy && metric.accuracy > 0.8 ? 'green' : 'blue'}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">{info.icon}</span>
                <Text className="font-semibold">{info.name}</Text>
              </div>
              <Text className="text-gray-400 text-sm mb-3">{info.description}</Text>

              {metric.type === 'regression' && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Text>MAE</Text>
                    <Text className="font-mono">{metric.mae}</Text>
                  </div>
                  <div className="flex justify-between items-center">
                    <Text>R² Score</Text>
                    <MetricBadge value={metric.r2 || 0} type="r2" />
                  </div>
                </div>
              )}

              {metric.type === 'classification' && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Text>Accuracy</Text>
                    <MetricBadge value={metric.accuracy || 0} type="accuracy" />
                  </div>
                  <div className="flex justify-between items-center">
                    <Text>AUC-ROC</Text>
                    <MetricBadge value={metric.auc || 0} type="auc" />
                  </div>
                </div>
              )}

              {metric.type === 'timeseries' && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Text>MAE</Text>
                    <Text className="font-mono">{metric.mae}%</Text>
                  </div>
                  <div className="flex justify-between items-center">
                    <Text>R² Score</Text>
                    <MetricBadge value={metric.r2 || 0} type="r2" />
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </Grid>

      {/* Feature Importance */}
      <Grid numItemsMd={2} className="gap-6">
        <Card>
          <Title>ED Wait Time - Feature Importance</Title>
          <Text className="mb-4">Top factors affecting wait time predictions</Text>
          <BarList
            data={importance.ed_wait_time?.map(f => ({
              name: f.feature.replace(/_/g, ' '),
              value: Math.round(f.importance * 100),
            })) || []}
            color="blue"
          />
        </Card>

        <Card>
          <Title>Mortality Risk - Feature Importance</Title>
          <Text className="mb-4">Top factors affecting risk classification</Text>
          <BarList
            data={importance.mortality_risk?.map(f => ({
              name: f.feature.replace(/_/g, ' '),
              value: Math.round(f.importance * 100),
            })) || []}
            color="rose"
          />
        </Card>
      </Grid>

      {/* Training Info */}
      <Card>
        <Title>Training Configuration</Title>
        <Grid numItemsMd={3} className="gap-4 mt-4">
          <div>
            <Text className="text-gray-400">Algorithm</Text>
            <Metric>XGBoost</Metric>
          </div>
          <div>
            <Text className="text-gray-400">Training Samples</Text>
            <Metric>3,000</Metric>
          </div>
          <div>
            <Text className="text-gray-400">Test Split</Text>
            <Metric>20%</Metric>
          </div>
        </Grid>

        <div className="mt-6 space-y-3">
          <div>
            <div className="flex justify-between mb-1">
              <Text>ED Wait Time Model</Text>
              <Text>R² = 57.3%</Text>
            </div>
            <ProgressBar value={57.3} color="blue" />
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <Text>Hospital Load Forecaster</Text>
              <Text>R² = 74.3%</Text>
            </div>
            <ProgressBar value={74.3} color="green" />
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <Text>Mortality Risk Classifier</Text>
              <Text>AUC = 82.5%</Text>
            </div>
            <ProgressBar value={82.5} color="rose" />
          </div>
        </div>
      </Card>
    </div>
  );
}

'use client';

import { useState } from 'react';
import {
  Card,
  Title,
  Text,
  NumberInput,
  Select,
  SelectItem,
  Button,
  Metric,
  Flex,
  Badge,
  Divider,
} from '@tremor/react';
import { api } from '@/lib/api';
import type { PredictionResponse, ExplanationResponse } from '@/lib/types';

export default function PredictPage() {
  const [formData, setFormData] = useState({
    admission_type: 'EMERGENCY',
    hour: 14,
    admission_location: 'EMERGENCY ROOM ADMIT',
    ethnicity: 'WHITE',
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      const [predResult, explainResult] = await Promise.all([
        api.predictWaitTime(formData),
        api.explainPrediction(formData),
      ]);

      setPrediction(predResult);
      setExplanation(explainResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <Title>ED Wait Time Prediction</Title>
        <Text>Predict emergency department wait times with explainable AI</Text>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <Card>
          <Title>Patient Parameters</Title>

          <div className="mt-6 space-y-4">
            <div>
              <Text className="mb-2">Admission Type</Text>
              <Select
                value={formData.admission_type}
                onValueChange={(value) =>
                  setFormData({ ...formData, admission_type: value })
                }
              >
                <SelectItem value="EMERGENCY">Emergency</SelectItem>
                <SelectItem value="URGENT">Urgent</SelectItem>
                <SelectItem value="ELECTIVE">Elective</SelectItem>
              </Select>
            </div>

            <div>
              <Text className="mb-2">Hour of Day (0-23)</Text>
              <NumberInput
                value={formData.hour}
                onValueChange={(value) =>
                  setFormData({ ...formData, hour: value || 0 })
                }
                min={0}
                max={23}
              />
            </div>

            <div>
              <Text className="mb-2">Admission Location</Text>
              <Select
                value={formData.admission_location}
                onValueChange={(value) =>
                  setFormData({ ...formData, admission_location: value })
                }
              >
                <SelectItem value="EMERGENCY ROOM ADMIT">Emergency Room</SelectItem>
                <SelectItem value="CLINIC REFERRAL/PREMATURE">Clinic Referral</SelectItem>
                <SelectItem value="TRANSFER FROM HOSP/EXTRAM">Hospital Transfer</SelectItem>
                <SelectItem value="PHYS REFERRAL/NORMAL DELI">Physician Referral</SelectItem>
              </Select>
            </div>

            <div>
              <Text className="mb-2">Ethnicity</Text>
              <Select
                value={formData.ethnicity}
                onValueChange={(value) =>
                  setFormData({ ...formData, ethnicity: value })
                }
              >
                <SelectItem value="WHITE">White</SelectItem>
                <SelectItem value="BLACK/AFRICAN AMERICAN">Black/African American</SelectItem>
                <SelectItem value="HISPANIC OR LATINO">Hispanic/Latino</SelectItem>
                <SelectItem value="ASIAN">Asian</SelectItem>
                <SelectItem value="OTHER">Other</SelectItem>
                <SelectItem value="UNKNOWN/NOT SPECIFIED">Unknown</SelectItem>
              </Select>
            </div>

            <Button
              onClick={handlePredict}
              loading={loading}
              size="lg"
              className="w-full mt-6"
            >
              Predict Wait Time
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-red-900/20 border border-red-500 rounded-lg">
                <Text className="text-red-400">{error}</Text>
              </div>
            )}
          </div>
        </Card>

        {/* Results */}
        <div className="space-y-6">
          {prediction && (
            <Card decoration="top" decorationColor="blue">
              <Title>Prediction Result</Title>
              <div className="mt-4">
                <Metric>{prediction.predicted_wait_time.toFixed(0)} minutes</Metric>
                <Flex className="mt-4" justifyContent="start">
                  <Badge color="blue" size="lg">
                    95% CI: {prediction.confidence_interval.lower.toFixed(0)} - {prediction.confidence_interval.upper.toFixed(0)} min
                  </Badge>
                </Flex>
              </div>
            </Card>
          )}

          {explanation && (
            <Card>
              <Title>SHAP Explanation</Title>
              <Text className="mt-2">
                How each feature contributes to this prediction
              </Text>

              {explanation.base_value && (
                <div className="mt-4 p-3 bg-gray-800 rounded-lg">
                  <Text className="text-sm">
                    Base Value: {explanation.base_value.toFixed(1)} minutes
                  </Text>
                  <Text className="text-xs text-gray-400">
                    Average prediction across all patients
                  </Text>
                </div>
              )}

              <Divider />

              <Title className="text-sm mt-4">Feature Contributions</Title>
              <div className="mt-4 space-y-3">
                {explanation.top_contributors.slice(0, 8).map((contrib, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <Text className="text-sm truncate max-w-[200px]">
                      {contrib.feature.replace(/_/g, ' ')}
                    </Text>
                    <Flex justifyContent="end" className="gap-2">
                      <Badge
                        color={contrib.shap_value > 0 ? 'red' : 'emerald'}
                        size="sm"
                      >
                        {contrib.shap_value > 0 ? '+' : ''}{contrib.shap_value.toFixed(2)}
                      </Badge>
                      <Text className="text-xs text-gray-400 w-20 text-right">
                        {contrib.direction}
                      </Text>
                    </Flex>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
                <Text className="text-xs text-gray-400">
                  <strong>Interpretation:</strong> Red values increase the predicted wait time,
                  green values decrease it. The magnitude shows how much each feature
                  contributes to pushing the prediction away from the base value.
                </Text>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

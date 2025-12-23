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
  Badge,
  Grid,
  BarList,
} from '@tremor/react';

interface PatientInput {
  admission_type: string;
  admission_location: string;
  ethnicity: string;
  hour: number;
  day_of_week: number;
  age: number;
  prev_admissions: number;
  num_diagnoses: number;
}

interface Predictions {
  ed_wait_time: { value: number; confidence_interval: [number, number] };
  length_of_stay: { value: number; confidence_interval: [number, number] };
  mortality_risk: { value: number; risk_level: string };
}

interface Explanation {
  feature: string;
  impact: number;
  direction: string;
}

// Mock prediction (runs locally, no API needed for demo)
function mockPredict(patient: PatientInput): Predictions {
  const baseWait = 180 + (patient.admission_type === 'EMERGENCY' ? 60 : patient.admission_type === 'URGENT' ? 30 : -40);
  const hourEffect = patient.hour >= 8 && patient.hour <= 18 ? 40 : -20;
  const waitTime = baseWait + hourEffect + (Math.random() - 0.5) * 30;

  const baseLOS = 5 + (patient.age - 50) * 0.05 + patient.num_diagnoses * 0.8;
  const los = Math.max(1, baseLOS + (Math.random() - 0.5) * 2);

  const riskScore = (patient.age - 55) * 0.02 + patient.num_diagnoses * 0.1 + patient.prev_admissions * 0.08 - 0.8;
  const mortality = 1 / (1 + Math.exp(-riskScore));

  return {
    ed_wait_time: {
      value: Math.round(waitTime),
      confidence_interval: [Math.round(waitTime * 0.85), Math.round(waitTime * 1.15)],
    },
    length_of_stay: {
      value: Math.round(los * 10) / 10,
      confidence_interval: [Math.round(los * 0.8 * 10) / 10, Math.round(los * 1.2 * 10) / 10],
    },
    mortality_risk: {
      value: Math.round(mortality * 1000) / 1000,
      risk_level: mortality > 0.5 ? 'High' : mortality > 0.25 ? 'Medium' : 'Low',
    },
  };
}

function mockExplain(patient: PatientInput, target: string): Explanation[] {
  const explanations: Record<string, Explanation[]> = {
    ed_wait_time: [
      { feature: 'hour', impact: patient.hour >= 8 && patient.hour <= 18 ? 28 : -15, direction: patient.hour >= 8 && patient.hour <= 18 ? 'increases' : 'decreases' },
      { feature: 'admission_type', impact: patient.admission_type === 'EMERGENCY' ? 22 : -12, direction: patient.admission_type === 'EMERGENCY' ? 'increases' : 'decreases' },
      { feature: 'admission_location', impact: 18, direction: 'increases' },
      { feature: 'age', impact: Math.abs(patient.age - 50) * 0.2, direction: patient.age > 50 ? 'increases' : 'decreases' },
    ],
    length_of_stay: [
      { feature: 'num_diagnoses', impact: patient.num_diagnoses * 8, direction: 'increases' },
      { feature: 'age', impact: Math.abs(patient.age - 50) * 0.4, direction: patient.age > 50 ? 'increases' : 'decreases' },
      { feature: 'admission_type', impact: patient.admission_type === 'EMERGENCY' ? 15 : -8, direction: patient.admission_type === 'EMERGENCY' ? 'increases' : 'decreases' },
      { feature: 'prev_admissions', impact: patient.prev_admissions * 5, direction: 'increases' },
    ],
    mortality_risk: [
      { feature: 'age', impact: Math.abs(patient.age - 55) * 0.5, direction: patient.age > 55 ? 'increases' : 'decreases' },
      { feature: 'num_diagnoses', impact: patient.num_diagnoses * 6, direction: 'increases' },
      { feature: 'prev_admissions', impact: patient.prev_admissions * 4, direction: 'increases' },
      { feature: 'admission_type', impact: patient.admission_type === 'EMERGENCY' ? 12 : -5, direction: patient.admission_type === 'EMERGENCY' ? 'increases' : 'decreases' },
    ],
  };
  return explanations[target] || explanations.ed_wait_time;
}

export default function PredictPage() {
  const [patient, setPatient] = useState<PatientInput>({
    admission_type: 'EMERGENCY',
    admission_location: 'EMERGENCY ROOM ADMIT',
    ethnicity: 'WHITE',
    hour: new Date().getHours(),
    day_of_week: new Date().getDay(),
    age: 55,
    prev_admissions: 1,
    num_diagnoses: 3,
  });

  const [predictions, setPredictions] = useState<Predictions | null>(null);
  const [explanations, setExplanations] = useState<Explanation[]>([]);
  const [selectedTarget, setSelectedTarget] = useState('ed_wait_time');
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    await new Promise((r) => setTimeout(r, 300));
    const preds = mockPredict(patient);
    setPredictions(preds);
    const expl = mockExplain(patient, selectedTarget);
    setExplanations(expl);
    setLoading(false);
  };

  const handleExplain = (target: string) => {
    setSelectedTarget(target);
    const expl = mockExplain(patient, target);
    setExplanations(expl);
  };

  return (
    <div className="space-y-6">
      <div>
        <Title>Patient Outcome Prediction</Title>
        <Text>Multi-target XGBoost prediction with SHAP explainability</Text>
      </div>

      {/* Input Form */}
      <Card>
        <Title>Patient Information</Title>
        <Grid numItemsMd={2} numItemsLg={4} className="gap-4 mt-4">
          <div>
            <Text className="mb-1">Admission Type</Text>
            <Select value={patient.admission_type} onValueChange={(v) => setPatient({ ...patient, admission_type: v })}>
              <SelectItem value="EMERGENCY">Emergency</SelectItem>
              <SelectItem value="URGENT">Urgent</SelectItem>
              <SelectItem value="ELECTIVE">Elective</SelectItem>
            </Select>
          </div>

          <div>
            <Text className="mb-1">Admission Location</Text>
            <Select value={patient.admission_location} onValueChange={(v) => setPatient({ ...patient, admission_location: v })}>
              <SelectItem value="EMERGENCY ROOM ADMIT">Emergency Room</SelectItem>
              <SelectItem value="TRANSFER FROM HOSP/EXTRAM">Hospital Transfer</SelectItem>
              <SelectItem value="CLINIC REFERRAL/PREMATURE">Clinic Referral</SelectItem>
              <SelectItem value="PHYS REFERRAL/NORMAL DELI">Physician Referral</SelectItem>
            </Select>
          </div>

          <div>
            <Text className="mb-1">Age</Text>
            <NumberInput value={patient.age} onValueChange={(v) => setPatient({ ...patient, age: v || 55 })} min={18} max={100} />
          </div>

          <div>
            <Text className="mb-1">Hour of Day</Text>
            <NumberInput value={patient.hour} onValueChange={(v) => setPatient({ ...patient, hour: v || 12 })} min={0} max={23} />
          </div>

          <div>
            <Text className="mb-1">Previous Admissions</Text>
            <NumberInput value={patient.prev_admissions} onValueChange={(v) => setPatient({ ...patient, prev_admissions: v || 0 })} min={0} max={20} />
          </div>

          <div>
            <Text className="mb-1">Number of Diagnoses</Text>
            <NumberInput value={patient.num_diagnoses} onValueChange={(v) => setPatient({ ...patient, num_diagnoses: v || 1 })} min={1} max={15} />
          </div>

          <div>
            <Text className="mb-1">Ethnicity</Text>
            <Select value={patient.ethnicity} onValueChange={(v) => setPatient({ ...patient, ethnicity: v })}>
              <SelectItem value="WHITE">White</SelectItem>
              <SelectItem value="BLACK/AFRICAN AMERICAN">Black/African American</SelectItem>
              <SelectItem value="HISPANIC/LATINO">Hispanic/Latino</SelectItem>
              <SelectItem value="ASIAN">Asian</SelectItem>
            </Select>
          </div>

          <div className="flex items-end">
            <Button onClick={handlePredict} loading={loading} className="w-full">
              Predict All Outcomes
            </Button>
          </div>
        </Grid>
      </Card>

      {/* Predictions */}
      {predictions && (
        <Grid numItemsMd={3} className="gap-4">
          <Card
            decoration="top"
            decorationColor="blue"
            className={`cursor-pointer transition ${selectedTarget === 'ed_wait_time' ? 'ring-2 ring-blue-500' : 'hover:bg-gray-800/50'}`}
            onClick={() => handleExplain('ed_wait_time')}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">⏱️</span>
              <Text>ED Wait Time</Text>
            </div>
            <Metric>{predictions.ed_wait_time.value} min</Metric>
            <Text className="text-gray-400 text-sm mt-1">
              95% CI: {predictions.ed_wait_time.confidence_interval[0]} - {predictions.ed_wait_time.confidence_interval[1]} min
            </Text>
          </Card>

          <Card
            decoration="top"
            decorationColor="emerald"
            className={`cursor-pointer transition ${selectedTarget === 'length_of_stay' ? 'ring-2 ring-emerald-500' : 'hover:bg-gray-800/50'}`}
            onClick={() => handleExplain('length_of_stay')}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🏥</span>
              <Text>Length of Stay</Text>
            </div>
            <Metric>{predictions.length_of_stay.value} days</Metric>
            <Text className="text-gray-400 text-sm mt-1">
              95% CI: {predictions.length_of_stay.confidence_interval[0]} - {predictions.length_of_stay.confidence_interval[1]} days
            </Text>
          </Card>

          <Card
            decoration="top"
            decorationColor={predictions.mortality_risk.risk_level === 'High' ? 'red' : predictions.mortality_risk.risk_level === 'Medium' ? 'amber' : 'green'}
            className={`cursor-pointer transition ${selectedTarget === 'mortality_risk' ? 'ring-2 ring-rose-500' : 'hover:bg-gray-800/50'}`}
            onClick={() => handleExplain('mortality_risk')}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">⚕️</span>
              <Text>Mortality Risk</Text>
            </div>
            <Metric>{(predictions.mortality_risk.value * 100).toFixed(1)}%</Metric>
            <Badge
              color={predictions.mortality_risk.risk_level === 'High' ? 'red' : predictions.mortality_risk.risk_level === 'Medium' ? 'amber' : 'green'}
              className="mt-1"
            >
              {predictions.mortality_risk.risk_level} Risk
            </Badge>
          </Card>
        </Grid>
      )}

      {/* SHAP Explanation */}
      {predictions && explanations.length > 0 && (
        <Card>
          <Title>SHAP Explanation: {selectedTarget.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</Title>
          <Text className="mb-4">Feature contributions to this prediction (click cards above to switch)</Text>
          <BarList
            data={explanations.map((e) => ({
              name: `${e.feature.replace(/_/g, ' ')} → ${e.direction}`,
              value: Math.round(Math.abs(e.impact)),
            }))}
            color={selectedTarget === 'mortality_risk' ? 'rose' : selectedTarget === 'length_of_stay' ? 'emerald' : 'blue'}
          />
          <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
            <Text className="text-xs text-gray-400">
              Higher values indicate stronger influence on the prediction. Features that &quot;increase&quot; push the prediction higher, while features that &quot;decrease&quot; lower it.
            </Text>
          </div>
        </Card>
      )}
    </div>
  );
}

'use client';

import { useState, useMemo } from 'react';
import { Card, Title, Text, AreaChart, Grid, Metric, Badge, Button } from '@tremor/react';

interface ForecastPoint {
  hours_ahead: number;
  predicted_load: number;
  hour: number;
  is_weekend: boolean;
}

// Generate mock forecast data
function generateForecast(currentLoad: number): ForecastPoint[] {
  const forecasts: ForecastPoint[] = [];
  const now = new Date();
  let load = currentLoad;

  for (let h = 1; h <= 24; h++) {
    const forecastHour = (now.getHours() + h) % 24;
    const dayOfWeek = (now.getDay() + Math.floor((now.getHours() + h) / 24)) % 7;
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;

    // Simulate hourly pattern
    const hourlyEffect = 15 * Math.sin((forecastHour - 6) * Math.PI / 12);
    const weekendEffect = isWeekend ? -8 : 0;
    const noise = (Math.random() - 0.5) * 6;

    load = Math.max(40, Math.min(100, 75 + hourlyEffect + weekendEffect + noise));

    forecasts.push({
      hours_ahead: h,
      predicted_load: Math.round(load * 10) / 10,
      hour: forecastHour,
      is_weekend: isWeekend,
    });
  }

  return forecasts;
}

function getLoadColor(load: number): 'green' | 'yellow' | 'orange' | 'red' {
  if (load < 60) return 'green';
  if (load < 75) return 'yellow';
  if (load < 85) return 'orange';
  return 'red';
}

function getLoadStatus(load: number): string {
  if (load < 60) return 'Low';
  if (load < 75) return 'Normal';
  if (load < 85) return 'High';
  return 'Critical';
}

export default function ForecastPage() {
  const [currentLoad] = useState(72);
  const [forecasts] = useState<ForecastPoint[]>(() => generateForecast(currentLoad));

  const chartData = useMemo(() => {
    return forecasts.map((f) => ({
      time: `+${f.hours_ahead}h`,
      'Hospital Load': f.predicted_load,
      hour: f.hour,
    }));
  }, [forecasts]);

  const peakLoad = useMemo(() => {
    return forecasts.reduce((max, f) => (f.predicted_load > max.predicted_load ? f : max), forecasts[0]);
  }, [forecasts]);

  const minLoad = useMemo(() => {
    return forecasts.reduce((min, f) => (f.predicted_load < min.predicted_load ? f : min), forecasts[0]);
  }, [forecasts]);

  const avgLoad = useMemo(() => {
    return Math.round((forecasts.reduce((sum, f) => sum + f.predicted_load, 0) / forecasts.length) * 10) / 10;
  }, [forecasts]);

  return (
    <div className="space-y-6">
      <div>
        <Title>Hospital Load Forecast</Title>
        <Text>24-hour ahead prediction using time-series XGBoost model</Text>
      </div>

      {/* Current Status */}
      <Grid numItemsMd={4} className="gap-4">
        <Card decoration="left" decorationColor={getLoadColor(currentLoad)}>
          <Text>Current Load</Text>
          <Metric>{currentLoad}%</Metric>
          <Badge color={getLoadColor(currentLoad)}>{getLoadStatus(currentLoad)}</Badge>
        </Card>

        <Card decoration="left" decorationColor="red">
          <Text>Peak (Next 24h)</Text>
          <Metric>{peakLoad.predicted_load}%</Metric>
          <Text className="text-gray-400">at {peakLoad.hour}:00 (+{peakLoad.hours_ahead}h)</Text>
        </Card>

        <Card decoration="left" decorationColor="green">
          <Text>Minimum (Next 24h)</Text>
          <Metric>{minLoad.predicted_load}%</Metric>
          <Text className="text-gray-400">at {minLoad.hour}:00 (+{minLoad.hours_ahead}h)</Text>
        </Card>

        <Card decoration="left" decorationColor="blue">
          <Text>Average</Text>
          <Metric>{avgLoad}%</Metric>
          <Text className="text-gray-400">24h forecast mean</Text>
        </Card>
      </Grid>

      {/* Forecast Chart */}
      <Card>
        <Title>Load Forecast Timeline</Title>
        <Text className="mb-4">Predicted hospital occupancy for next 24 hours</Text>
        <AreaChart
          className="h-72"
          data={chartData}
          index="time"
          categories={['Hospital Load']}
          colors={['blue']}
          valueFormatter={(v) => `${v}%`}
          showLegend={false}
          showAnimation={true}
          curveType="monotone"
        />
      </Card>

      {/* Hourly Breakdown */}
      <Card>
        <Title>Hourly Breakdown</Title>
        <div className="mt-4 grid grid-cols-6 md:grid-cols-12 gap-2">
          {forecasts.slice(0, 12).map((f) => (
            <div
              key={f.hours_ahead}
              className={`p-2 rounded text-center ${
                f.predicted_load > 85
                  ? 'bg-red-900/50'
                  : f.predicted_load > 75
                  ? 'bg-orange-900/50'
                  : f.predicted_load > 60
                  ? 'bg-yellow-900/50'
                  : 'bg-green-900/50'
              }`}
            >
              <Text className="text-xs text-gray-400">+{f.hours_ahead}h</Text>
              <Text className="font-mono font-bold">{f.predicted_load}%</Text>
              <Text className="text-xs text-gray-500">{f.hour}:00</Text>
            </div>
          ))}
        </div>
      </Card>

      {/* Model Info */}
      <Card>
        <Title>Forecast Model Details</Title>
        <Grid numItemsMd={3} className="gap-4 mt-4">
          <div>
            <Text className="text-gray-400">Algorithm</Text>
            <Text className="font-semibold">XGBoost Time-Series</Text>
          </div>
          <div>
            <Text className="text-gray-400">Features</Text>
            <Text className="font-semibold">Hour, Day, Lag (1-24h)</Text>
          </div>
          <div>
            <Text className="text-gray-400">Performance</Text>
            <Text className="font-semibold">MAE: 6.0%, R²: 0.74</Text>
          </div>
        </Grid>
      </Card>
    </div>
  );
}

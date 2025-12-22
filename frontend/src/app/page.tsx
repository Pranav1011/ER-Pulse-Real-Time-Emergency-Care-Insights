'use client';

import { useEffect, useState } from 'react';
import {
  Card,
  Title,
  Text,
  Grid,
  Col,
  Flex,
  Badge,
} from '@tremor/react';
import { MetricCard } from '@/components/MetricCard';
import { api } from '@/lib/api';
import type { MetricsResponse, AnomaliesResponse } from '@/lib/types';

export default function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [anomalies, setAnomalies] = useState<AnomaliesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [metricsData, anomaliesData] = await Promise.all([
          api.getCurrentMetrics(),
          api.getActiveAnomalies(),
        ]);
        setMetrics(metricsData);
        setAnomalies(anomaliesData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Text>Loading metrics...</Text>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/20 border-red-500">
        <Title>Error</Title>
        <Text>{error}</Text>
        <Text className="mt-2 text-sm">
          Make sure the backend API is running at {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
        </Text>
      </Card>
    );
  }

  const metricsList = metrics?.metrics ? Object.entries(metrics.metrics) : [];
  const systemStatus = anomalies?.is_system_anomalous ? 'Anomalous' : 'Normal';
  const statusColor = anomalies?.is_system_anomalous ? 'red' : 'emerald';

  return (
    <div className="space-y-8">
      {/* Header */}
      <Flex justifyContent="between" alignItems="center">
        <div>
          <Title>Live Metrics Dashboard</Title>
          <Text>Real-time hospital operations monitoring</Text>
        </div>
        <Flex className="gap-4" alignItems="center">
          <Text className="text-sm text-gray-400">
            Last updated: {metrics?.timestamp ? new Date(metrics.timestamp).toLocaleTimeString() : 'N/A'}
          </Text>
          <Badge color={statusColor} size="lg">
            System: {systemStatus}
          </Badge>
        </Flex>
      </Flex>

      {/* Multi-dimensional Score */}
      {anomalies && (
        <Card decoration="left" decorationColor={statusColor}>
          <Flex>
            <div>
              <Text>Multi-Dimensional Anomaly Score</Text>
              <Title className="text-3xl">{anomalies.multi_dimensional_score.toFixed(2)}</Title>
            </div>
            <div className="text-right">
              <Text>Active Alerts</Text>
              <Title className="text-3xl">{anomalies.active_anomalies.length}</Title>
            </div>
          </Flex>
        </Card>
      )}

      {/* Metric Cards Grid */}
      <Grid numItemsMd={2} numItemsLg={3} className="gap-6">
        {metricsList.map(([key, value]) => (
          <Col key={key}>
            <MetricCard
              title={key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              metric={value}
              unit={key === 'ed_wait_time' ? 'min' : key === 'length_of_stay' ? 'hrs' : ''}
            />
          </Col>
        ))}
      </Grid>

      {/* Active Anomalies List */}
      {anomalies && anomalies.active_anomalies.length > 0 && (
        <Card>
          <Title>Active Anomalies</Title>
          <div className="mt-4 space-y-3">
            {anomalies.active_anomalies.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border ${
                  alert.severity === 'critical'
                    ? 'bg-red-900/20 border-red-500'
                    : 'bg-yellow-900/20 border-yellow-500'
                }`}
              >
                <Flex justifyContent="between">
                  <div>
                    <Text className="font-semibold">
                      {alert.metric.replace(/_/g, ' ').toUpperCase()}
                    </Text>
                    <Text className="text-sm text-gray-400">{alert.context}</Text>
                  </div>
                  <div className="text-right">
                    <Text>Current: {alert.current.toFixed(1)}</Text>
                    <Text className="text-sm text-gray-400">
                      Baseline: {alert.baseline.toFixed(1)} | Z: {alert.z_score.toFixed(2)}
                    </Text>
                  </div>
                </Flex>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

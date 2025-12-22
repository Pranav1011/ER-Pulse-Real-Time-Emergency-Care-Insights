'use client';

import { useEffect, useState } from 'react';
import {
  Card,
  Title,
  Text,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Badge,
  Flex,
} from '@tremor/react';
import { AnomalyBadge } from '@/components/AnomalyBadge';
import { api } from '@/lib/api';
import type { AnomaliesResponse } from '@/lib/types';

export default function AlertsPage() {
  const [anomalies, setAnomalies] = useState<AnomaliesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await api.getActiveAnomalies();
        setAnomalies(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch anomalies');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Text>Loading anomalies...</Text>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/20 border-red-500">
        <Title>Error</Title>
        <Text>{error}</Text>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <Title>Anomaly Alerts</Title>
        <Text>Real-time detection of unusual patterns in hospital operations</Text>
      </div>

      {/* Summary Card */}
      <Card>
        <Flex justifyContent="between" alignItems="center">
          <div>
            <Text>System Status</Text>
            <Title className="text-2xl">
              {anomalies?.is_system_anomalous ? 'Anomalous Activity Detected' : 'Operating Normally'}
            </Title>
          </div>
          <Badge
            color={anomalies?.is_system_anomalous ? 'red' : 'emerald'}
            size="xl"
          >
            Score: {anomalies?.multi_dimensional_score.toFixed(2)}
          </Badge>
        </Flex>
      </Card>

      {/* Alerts Table */}
      <Card>
        <Title>Active Alerts</Title>
        {anomalies && anomalies.active_anomalies.length > 0 ? (
          <Table className="mt-4">
            <TableHead>
              <TableRow>
                <TableHeaderCell>Timestamp</TableHeaderCell>
                <TableHeaderCell>Metric</TableHeaderCell>
                <TableHeaderCell>Current</TableHeaderCell>
                <TableHeaderCell>Baseline</TableHeaderCell>
                <TableHeaderCell>Z-Score</TableHeaderCell>
                <TableHeaderCell>Severity</TableHeaderCell>
                <TableHeaderCell>Context</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {anomalies.active_anomalies.map((alert) => (
                <TableRow key={alert.id}>
                  <TableCell>
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </TableCell>
                  <TableCell className="font-medium">
                    {alert.metric.replace(/_/g, ' ')}
                  </TableCell>
                  <TableCell>{alert.current.toFixed(1)}</TableCell>
                  <TableCell>{alert.baseline.toFixed(1)}</TableCell>
                  <TableCell>
                    <Badge color={Math.abs(alert.z_score) > 3 ? 'red' : 'yellow'}>
                      {alert.z_score.toFixed(2)}σ
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <AnomalyBadge severity={alert.severity} />
                  </TableCell>
                  <TableCell className="text-sm text-gray-400">
                    {alert.context}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="mt-8 text-center py-12">
            <Text className="text-xl">No Active Anomalies</Text>
            <Text className="mt-2 text-gray-400">
              All metrics are within expected ranges
            </Text>
          </div>
        )}
      </Card>

      {/* Explanation Card */}
      <Card>
        <Title>How Anomaly Detection Works</Title>
        <div className="mt-4 space-y-4 text-sm text-gray-300">
          <div>
            <Text className="font-semibold text-white">Seasonal Baselines</Text>
            <Text>
              Each metric is compared against historical patterns for the same hour and day of week.
              For example, Monday 2PM is compared to previous Monday 2PM values.
            </Text>
          </div>
          <div>
            <Text className="font-semibold text-white">Z-Score Thresholds</Text>
            <Text>
              |Z| &gt; 2.5: Warning - metric is significantly different from baseline
            </Text>
            <Text>
              |Z| &gt; 3.0: Critical - metric is extremely unusual
            </Text>
          </div>
          <div>
            <Text className="font-semibold text-white">Multi-Dimensional Detection</Text>
            <Text>
              Isolation Forest algorithm detects unusual combinations of metrics,
              even when individual metrics appear normal.
            </Text>
          </div>
        </div>
      </Card>
    </div>
  );
}

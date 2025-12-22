'use client';

import { Card, Metric, Text, Flex, BadgeDelta, ProgressBar } from '@tremor/react';
import type { MetricValue } from '@/lib/types';

interface MetricCardProps {
  title: string;
  metric: MetricValue;
  unit?: string;
}

function getSeverityColor(severity: string): 'emerald' | 'yellow' | 'red' {
  switch (severity) {
    case 'critical':
      return 'red';
    case 'warning':
      return 'yellow';
    default:
      return 'emerald';
  }
}

function getDeltaType(zScore: number): 'increase' | 'decrease' | 'unchanged' {
  if (Math.abs(zScore) < 0.5) return 'unchanged';
  return zScore > 0 ? 'increase' : 'decrease';
}

export function MetricCard({ title, metric, unit = '' }: MetricCardProps) {
  const color = getSeverityColor(metric.severity);
  const deltaType = getDeltaType(metric.z_score);
  const percentFromBaseline = ((metric.current - metric.baseline) / metric.baseline) * 100;

  return (
    <Card className="max-w-sm" decoration="top" decorationColor={color}>
      <Flex justifyContent="between" alignItems="center">
        <Text>{title}</Text>
        <BadgeDelta deltaType={deltaType} size="xs">
          {metric.z_score > 0 ? '+' : ''}{metric.z_score.toFixed(1)}σ
        </BadgeDelta>
      </Flex>
      <Metric className="mt-2">
        {metric.current.toFixed(1)} {unit}
      </Metric>
      <Flex className="mt-2">
        <Text className="text-xs text-tremor-content-subtle">
          Baseline: {metric.baseline.toFixed(1)} {unit}
        </Text>
        <Text className="text-xs text-tremor-content-subtle">
          {percentFromBaseline > 0 ? '+' : ''}{percentFromBaseline.toFixed(0)}%
        </Text>
      </Flex>
      <ProgressBar
        value={Math.min(Math.abs(metric.z_score) * 20, 100)}
        color={color}
        className="mt-2"
      />
      <Text className="mt-2 text-xs text-tremor-content-subtle">
        {metric.context}
      </Text>
    </Card>
  );
}

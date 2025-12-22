'use client';

import { Badge } from '@tremor/react';

interface AnomalyBadgeProps {
  severity: 'info' | 'warning' | 'critical';
}

export function AnomalyBadge({ severity }: AnomalyBadgeProps) {
  const colors: Record<string, 'emerald' | 'yellow' | 'red'> = {
    info: 'emerald',
    warning: 'yellow',
    critical: 'red',
  };

  const labels: Record<string, string> = {
    info: 'Normal',
    warning: 'Warning',
    critical: 'Critical',
  };

  return (
    <Badge color={colors[severity]} size="sm">
      {labels[severity]}
    </Badge>
  );
}

'use client';

import { SparkAreaChart } from '@tremor/react';

interface SparklineChartProps {
  data: { hour: number; value: number }[];
  color?: 'blue' | 'emerald' | 'red' | 'yellow';
}

export function SparklineChart({ data, color = 'blue' }: SparklineChartProps) {
  return (
    <SparkAreaChart
      data={data}
      categories={['value']}
      index="hour"
      colors={[color]}
      className="h-10 w-36"
    />
  );
}

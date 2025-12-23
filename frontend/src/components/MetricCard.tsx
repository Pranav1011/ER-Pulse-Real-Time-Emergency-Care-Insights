'use client';

import { ArrowUp, ArrowDown, Minus, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';
import type { MetricValue } from '@/lib/types';

interface MetricCardProps {
  title: string;
  metric: MetricValue;
  unit?: string;
}

function getSeverityConfig(severity: string) {
  switch (severity) {
    case 'critical':
      return {
        bg: 'bg-gradient-to-br from-red-500/20 to-red-600/10',
        border: 'border-red-500/50',
        icon: AlertTriangle,
        iconColor: 'text-red-400',
        glow: 'shadow-red-500/20',
      };
    case 'warning':
      return {
        bg: 'bg-gradient-to-br from-amber-500/20 to-amber-600/10',
        border: 'border-amber-500/50',
        icon: AlertCircle,
        iconColor: 'text-amber-400',
        glow: 'shadow-amber-500/20',
      };
    default:
      return {
        bg: 'bg-gradient-to-br from-emerald-500/20 to-emerald-600/10',
        border: 'border-emerald-500/50',
        icon: CheckCircle,
        iconColor: 'text-emerald-400',
        glow: 'shadow-emerald-500/20',
      };
  }
}

function getZScoreDisplay(zScore: number) {
  const absZ = Math.abs(zScore);
  if (absZ < 0.5) {
    return { Icon: Minus, color: 'text-gray-400', bgColor: 'bg-gray-500/20' };
  }
  if (zScore > 0) {
    return { Icon: ArrowUp, color: 'text-red-400', bgColor: 'bg-red-500/20' };
  }
  return { Icon: ArrowDown, color: 'text-emerald-400', bgColor: 'bg-emerald-500/20' };
}

export function MetricCard({ title, metric, unit = '' }: MetricCardProps) {
  const config = getSeverityConfig(metric.severity);
  const zDisplay = getZScoreDisplay(metric.z_score);
  const StatusIcon = config.icon;

  // Fix infinity bug - handle zero baseline
  const percentFromBaseline = metric.baseline !== 0
    ? ((metric.current - metric.baseline) / metric.baseline) * 100
    : metric.current > 0 ? 100 : 0;

  const displayPercent = isFinite(percentFromBaseline)
    ? `${percentFromBaseline > 0 ? '+' : ''}${percentFromBaseline.toFixed(0)}%`
    : 'N/A';

  return (
    <div className={`relative overflow-hidden rounded-2xl border ${config.border} ${config.bg} p-6 shadow-lg ${config.glow} transition-all duration-300 hover:scale-[1.02] hover:shadow-xl`}>
      {/* Status indicator */}
      <div className="absolute top-4 right-4">
        <StatusIcon className={`h-5 w-5 ${config.iconColor}`} />
      </div>

      {/* Title */}
      <p className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-1">
        {title}
      </p>

      {/* Main Value */}
      <div className="flex items-baseline gap-2 mb-4">
        <span className="text-4xl font-bold text-white">
          {metric.current.toFixed(1)}
        </span>
        {unit && <span className="text-lg text-gray-400">{unit}</span>}
      </div>

      {/* Z-Score Badge */}
      <div className="flex items-center gap-3 mb-4">
        <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${zDisplay.bgColor}`}>
          <zDisplay.Icon className={`h-4 w-4 ${zDisplay.color}`} />
          <span className={`text-sm font-semibold ${zDisplay.color}`}>
            {metric.z_score > 0 ? '+' : ''}{metric.z_score.toFixed(1)}σ
          </span>
        </div>
        <span className="text-sm text-gray-500">
          {displayPercent}
        </span>
      </div>

      {/* Progress Bar */}
      <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden mb-3">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            metric.severity === 'critical' ? 'bg-gradient-to-r from-red-500 to-red-400' :
            metric.severity === 'warning' ? 'bg-gradient-to-r from-amber-500 to-amber-400' :
            'bg-gradient-to-r from-emerald-500 to-emerald-400'
          }`}
          style={{ width: `${Math.min(Math.abs(metric.z_score) * 20, 100)}%` }}
        />
      </div>

      {/* Baseline Info */}
      <div className="flex justify-between items-center text-xs text-gray-500">
        <span>Baseline: {metric.baseline.toFixed(1)} {unit}</span>
      </div>

      {/* Context */}
      <p className="mt-2 text-xs text-gray-500 truncate">
        {metric.context}
      </p>
    </div>
  );
}

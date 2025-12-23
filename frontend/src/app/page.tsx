'use client';

import { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  Clock,
  Users,
  Bed,
  Timer,
  RefreshCw,
  Shield,
  ShieldAlert,
} from 'lucide-react';
import { MetricCard } from '@/components/MetricCard';
import { api } from '@/lib/api';
import type { MetricsResponse, AnomaliesResponse } from '@/lib/types';

const metricIcons: Record<string, typeof Activity> = {
  ed_wait_time: Clock,
  admission_rate: Users,
  transfer_delay: Timer,
  department_load: Bed,
  length_of_stay: Activity,
};

const metricUnits: Record<string, string> = {
  ed_wait_time: 'min',
  admission_rate: '/hr',
  transfer_delay: 'hrs',
  department_load: '%',
  length_of_stay: 'hrs',
};

export default function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [anomalies, setAnomalies] = useState<AnomaliesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  async function fetchData(isRefresh = false) {
    if (isRefresh) setRefreshing(true);
    try {
      const [metricsData, anomaliesData] = await Promise.all([
        api.getCurrentMetrics(),
        api.getActiveAnomalies(),
      ]);
      setMetrics(metricsData);
      setAnomalies(anomaliesData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => fetchData(), 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-4 border-blue-500/20"></div>
            <div className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent animate-spin"></div>
          </div>
          <p className="text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-red-500/10 border border-red-500/50 rounded-2xl p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Connection Error</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <p className="text-sm text-gray-500 mb-6">
            Backend: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </p>
          <button
            onClick={() => { setLoading(true); fetchData(); }}
            className="px-6 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const metricsList = metrics?.metrics ? Object.entries(metrics.metrics) : [];
  const isAnomalous = anomalies?.is_system_anomalous ?? false;
  const criticalCount = anomalies?.active_anomalies.filter(a => a.severity === 'critical').length ?? 0;
  const warningCount = anomalies?.active_anomalies.filter(a => a.severity === 'warning').length ?? 0;

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Activity className="h-8 w-8 text-blue-400" />
            Live Metrics Dashboard
          </h1>
          <p className="text-gray-400 mt-1">
            Real-time hospital operations monitoring
          </p>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => fetchData(true)}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl text-gray-300 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <div className="text-sm text-gray-500">
            Updated: {metrics?.timestamp ? new Date(metrics.timestamp).toLocaleTimeString() : 'N/A'}
          </div>
        </div>
      </div>

      {/* Status Cards Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* System Status */}
        <div className={`rounded-2xl p-6 border ${
          isAnomalous
            ? 'bg-gradient-to-br from-red-500/20 to-red-600/5 border-red-500/50'
            : 'bg-gradient-to-br from-emerald-500/20 to-emerald-600/5 border-emerald-500/50'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400 uppercase tracking-wide">System Status</p>
              <p className={`text-2xl font-bold mt-1 ${isAnomalous ? 'text-red-400' : 'text-emerald-400'}`}>
                {isAnomalous ? 'Anomalous' : 'Normal'}
              </p>
            </div>
            {isAnomalous ? (
              <ShieldAlert className="h-12 w-12 text-red-400/50" />
            ) : (
              <Shield className="h-12 w-12 text-emerald-400/50" />
            )}
          </div>
        </div>

        {/* Anomaly Score */}
        <div className="rounded-2xl p-6 bg-gradient-to-br from-blue-500/20 to-blue-600/5 border border-blue-500/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400 uppercase tracking-wide">Anomaly Score</p>
              <p className="text-2xl font-bold text-blue-400 mt-1">
                {anomalies?.multi_dimensional_score.toFixed(1) ?? '0'}
              </p>
              <p className="text-xs text-gray-500 mt-1">Threshold: 5.0</p>
            </div>
            <TrendingUp className="h-12 w-12 text-blue-400/50" />
          </div>
        </div>

        {/* Active Alerts */}
        <div className="rounded-2xl p-6 bg-gradient-to-br from-amber-500/20 to-amber-600/5 border border-amber-500/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400 uppercase tracking-wide">Active Alerts</p>
              <p className="text-2xl font-bold text-amber-400 mt-1">
                {anomalies?.active_anomalies.length ?? 0}
              </p>
              <div className="flex gap-3 mt-1 text-xs">
                {criticalCount > 0 && (
                  <span className="text-red-400">{criticalCount} critical</span>
                )}
                {warningCount > 0 && (
                  <span className="text-amber-400">{warningCount} warning</span>
                )}
              </div>
            </div>
            <AlertTriangle className="h-12 w-12 text-amber-400/50" />
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div>
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-gray-400" />
          Key Metrics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          {metricsList.map(([key, value]) => (
            <MetricCard
              key={key}
              title={key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              metric={value}
              unit={metricUnits[key] || ''}
            />
          ))}
        </div>
      </div>

      {/* Active Anomalies Section */}
      {anomalies && anomalies.active_anomalies.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-400" />
            Active Anomalies
          </h2>
          <div className="space-y-3">
            {anomalies.active_anomalies.map((alert) => {
              const Icon = metricIcons[alert.metric] || Activity;
              const isCritical = alert.severity === 'critical';

              return (
                <div
                  key={alert.id}
                  className={`rounded-xl border p-5 transition-all hover:scale-[1.01] ${
                    isCritical
                      ? 'bg-gradient-to-r from-red-500/20 to-transparent border-red-500/50'
                      : 'bg-gradient-to-r from-amber-500/20 to-transparent border-amber-500/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`p-3 rounded-xl ${isCritical ? 'bg-red-500/20' : 'bg-amber-500/20'}`}>
                        <Icon className={`h-6 w-6 ${isCritical ? 'text-red-400' : 'text-amber-400'}`} />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">
                          {alert.metric.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                        </h3>
                        <p className="text-sm text-gray-400">{alert.context}</p>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="flex items-center gap-2 justify-end">
                        <span className="text-2xl font-bold text-white">{alert.current.toFixed(1)}</span>
                        <span className={`px-2 py-1 rounded-lg text-xs font-semibold ${
                          isCritical ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'
                        }`}>
                          {alert.z_score > 0 ? '+' : ''}{alert.z_score.toFixed(1)}σ
                        </span>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">
                        Baseline: {alert.baseline.toFixed(1)}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="text-center text-xs text-gray-600 pt-4 border-t border-gray-800">
        Data refreshes automatically every 30 seconds
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { MeasurementHistoryResponse, MeasurementTrend } from '@/lib/types';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, Calendar, Clock, Target, Activity } from 'lucide-react';

const TIME_PERIODS = [
  { label: '7 Days', value: 7 },
  { label: '30 Days', value: 30 },
  { label: '90 Days', value: 90 },
  { label: '1 Year', value: 365 },
];

export default function HistoryPage() {
  const [days, setDays] = useState(30);

  const historyQuery = useQuery<MeasurementHistoryResponse>({
    queryKey: ['history', days],
    queryFn: async () => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return brandAPI.getHistory(apiKey, days);
    },
  });

  if (historyQuery.isLoading) {
    return <HistoryPageSkeleton />;
  }

  if (historyQuery.error) {
    return (
      <div className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-3 rounded-lg">
        {(historyQuery.error as Error)?.message}
      </div>
    );
  }

  const history = historyQuery.data;
  if (!history) {
    return (
      <div className="bg-warning/10 border border-warning/20 text-warning-foreground px-4 py-3 rounded-lg">
        No history data available yet
      </div>
    );
  }

  // Format daily counts for charts
  const chartData = history.daily_counts.map((item) => ({
    date: new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    fullDate: item.date,
    count: item.count,
    confidence: Math.round(item.avg_confidence * 100),
    processingTime: Math.round(item.avg_processing_time),
  }));

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Measurement History</h1>
          <p className="text-muted-foreground mt-1">Track trends and patterns over time</p>
        </div>

        {/* Time Period Selector */}
        <div className="flex gap-2">
          {TIME_PERIODS.map((period) => (
            <button
              key={period.value}
              onClick={() => setDays(period.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                days === period.value
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              {period.label}
            </button>
          ))}
        </div>
      </div>

      {/* Trend Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {history.trends.map((trend) => (
          <TrendCard key={trend.metric} trend={trend} />
        ))}
      </div>

      {/* Measurements Over Time Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Measurements Over Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12 }}
                  className="text-muted-foreground"
                  interval="preserveStartEnd"
                />
                <YAxis tick={{ fontSize: 12 }} className="text-muted-foreground" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="hsl(var(--primary))"
                  fillOpacity={1}
                  fill="url(#colorCount)"
                  name="Measurements"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confidence Score Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5" />
              Confidence Score Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData.filter((d) => d.confidence > 0)}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12 }}
                    className="text-muted-foreground"
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    className="text-muted-foreground"
                    domain={[0, 100]}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                    formatter={(value) => [`${value}%`, 'Confidence']}
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="hsl(142, 76%, 36%)"
                    strokeWidth={2}
                    dot={false}
                    name="Confidence"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Processing Time Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Processing Time Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData.filter((d) => d.processingTime > 0)}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12 }}
                    className="text-muted-foreground"
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    className="text-muted-foreground"
                    tickFormatter={(v) => `${v}ms`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                    formatter={(value) => [`${value}ms`, 'Processing Time']}
                  />
                  <Line
                    type="monotone"
                    dataKey="processingTime"
                    stroke="hsl(217, 91%, 60%)"
                    strokeWidth={2}
                    dot={false}
                    name="Processing Time"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Measurements Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="w-5 h-5" />
            Recent Measurements
          </CardTitle>
        </CardHeader>
        <CardContent>
          {history.recent_measurements.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No measurements recorded yet
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Date
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Shoulder
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Chest
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Waist
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Hip
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Size
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">
                      Time
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {history.recent_measurements.map((measurement) => (
                    <tr
                      key={measurement.id}
                      className="border-b border-border/50 hover:bg-muted/50 transition-colors"
                    >
                      <td className="py-3 px-4 text-sm">
                        {new Date(measurement.created_at).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </td>
                      <td className="py-3 px-4 text-sm font-medium">
                        {measurement.shoulder_width.toFixed(1)}cm
                      </td>
                      <td className="py-3 px-4 text-sm font-medium">
                        {measurement.chest_width.toFixed(1)}cm
                      </td>
                      <td className="py-3 px-4 text-sm font-medium">
                        {measurement.waist_width.toFixed(1)}cm
                      </td>
                      <td className="py-3 px-4 text-sm font-medium">
                        {measurement.hip_width.toFixed(1)}cm
                      </td>
                      <td className="py-3 px-4">
                        {measurement.recommended_size ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                            {measurement.recommended_size}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-sm text-muted-foreground">
                        {measurement.processing_time_ms.toFixed(0)}ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Summary Stats */}
      <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-6 rounded-xl border border-primary/20">
        <h3 className="text-lg font-semibold text-foreground mb-4">Period Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-muted-foreground">Total Measurements</p>
            <p className="text-2xl font-bold text-foreground">{history.total_count}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Daily Average</p>
            <p className="text-2xl font-bold text-foreground">
              {(history.total_count / days).toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Period Start</p>
            <p className="text-lg font-medium text-foreground">
              {new Date(history.period_start).toLocaleDateString()}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Period End</p>
            <p className="text-lg font-medium text-foreground">
              {new Date(history.period_end).toLocaleDateString()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrendCard({ trend }: { trend: MeasurementTrend }) {
  const getTrendIcon = () => {
    switch (trend.trend) {
      case 'up':
        return <TrendingUp className="w-5 h-5 text-green-500" />;
      case 'down':
        return <TrendingDown className="w-5 h-5 text-red-500" />;
      default:
        return <Minus className="w-5 h-5 text-muted-foreground" />;
    }
  };

  const getTrendColor = () => {
    // For processing time, down is good
    if (trend.metric === 'Avg Processing Time') {
      return trend.trend === 'down' ? 'text-green-500' : trend.trend === 'up' ? 'text-red-500' : 'text-muted-foreground';
    }
    return trend.trend === 'up' ? 'text-green-500' : trend.trend === 'down' ? 'text-red-500' : 'text-muted-foreground';
  };

  const formatValue = (value: number) => {
    if (trend.metric === 'Avg Confidence') {
      return `${value.toFixed(1)}%`;
    }
    if (trend.metric === 'Avg Processing Time') {
      return `${value.toFixed(0)}ms`;
    }
    return value.toFixed(0);
  };

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{trend.metric}</p>
            <p className="text-2xl font-bold text-foreground mt-1">
              {formatValue(trend.current_value)}
            </p>
          </div>
          {getTrendIcon()}
        </div>
        <div className={`mt-2 text-sm ${getTrendColor()}`}>
          {trend.change_percentage > 0 ? '+' : ''}
          {trend.change_percentage.toFixed(1)}% vs previous period
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Previous: {formatValue(trend.previous_value)}
        </p>
      </CardContent>
    </Card>
  );
}

function HistoryPageSkeleton() {
  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-64 mt-2" />
        </div>
        <div className="flex gap-2">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-10 w-20" />
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardContent className="pt-6">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-16 mt-2" />
              <Skeleton className="h-4 w-32 mt-2" />
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[1, 2].map((i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-6 w-40" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-[250px] w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

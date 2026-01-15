'use client';

import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { Analytics } from '@/lib/types';
import { useQuery } from '@tanstack/react-query';
import { BarChart3, Target, TrendingUp, DollarSign, Package, Sparkles } from 'lucide-react';

export default function AnalyticsPage() {
  const analyticsQuery = useQuery<Analytics>({
    queryKey: ['analytics'],
    queryFn: async () => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return brandAPI.getAnalytics(apiKey);
    },
  });

  if (analyticsQuery.isLoading) {
    return (
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <div className="h-8 w-40 bg-gray-200 rounded animate-pulse" />
          <div className="h-4 w-64 mt-2 bg-gray-200 rounded animate-pulse" />
        </div>
        <div className="grid md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="h-4 w-28 bg-gray-200 rounded animate-pulse" />
              <div className="h-8 w-20 mt-3 bg-gray-200 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (analyticsQuery.error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
        {(analyticsQuery.error as Error)?.message}
      </div>
    );
  }

  const analytics = analyticsQuery.data;
  if (!analytics) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-lg">
        No analytics data available yet
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-500 mt-1">Track performance and ROI metrics</p>
      </div>

      {/* Key Metrics */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Total Measurements</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">
                {analytics.total_measurements.toLocaleString()}
              </p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-indigo-100 flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-indigo-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Average Confidence</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">
                {(analytics.average_confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center">
              <Target className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">ROI</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">
                {analytics.revenue_impact.roi_percentage.toFixed(1)}%
              </p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-amber-100 flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-amber-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Size Distribution */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Size Distribution</h2>
        <div className="space-y-4">
          {Object.entries(analytics.size_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([size, count]) => {
              const total = Object.values(analytics.size_distribution).reduce((sum, val) => sum + val, 0);
              const percentage = (count / total) * 100;
              return (
                <div key={size}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900">Size {size}</span>
                    <span className="text-sm text-gray-500">
                      {count} ({percentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-600 rounded-full transition-all duration-500"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Popular Products */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Popular Products</h2>
        {analytics.popular_products.length === 0 ? (
          <p className="text-gray-500">No product data available yet</p>
        ) : (
          <div className="space-y-3">
            {analytics.popular_products.map((product, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-900">{product.product_name}</span>
                </div>
                <span className="text-sm text-gray-500">{product.measurement_count} measurements</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Revenue Impact */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-2 mb-4">
          <DollarSign className="w-5 h-5" />
          <h2 className="text-lg font-semibold">Revenue Impact</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <p className="text-indigo-200 text-sm">Estimated Conversions</p>
            <p className="text-3xl font-bold mt-1">{analytics.revenue_impact.estimated_conversions}</p>
            <p className="text-indigo-200 text-xs mt-1">Additional sales from accurate sizing</p>
          </div>
          <div>
            <p className="text-indigo-200 text-sm">Returns Prevented</p>
            <p className="text-3xl font-bold mt-1">{analytics.revenue_impact.estimated_returns_prevented}</p>
            <p className="text-indigo-200 text-xs mt-1">Fewer returns due to wrong size</p>
          </div>
          <div>
            <p className="text-indigo-200 text-sm">ROI Percentage</p>
            <p className="text-3xl font-bold mt-1">{analytics.revenue_impact.roi_percentage.toFixed(1)}%</p>
            <p className="text-indigo-200 text-xs mt-1">Return on investment</p>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-5 h-5 text-indigo-600" />
          <h3 className="font-semibold text-gray-900">Key Insights</h3>
        </div>
        <ul className="space-y-2 text-sm text-gray-600">
          <li className="flex items-start gap-2">
            <span className="text-indigo-600">•</span>
            Most customers fit into size{' '}
            <strong className="text-gray-900">
              {Object.entries(analytics.size_distribution).sort(([, a], [, b]) => b - a)[0]?.[0] || 'N/A'}
            </strong>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600">•</span>
            Average measurement confidence:{' '}
            <strong className="text-gray-900">{(analytics.average_confidence * 100).toFixed(1)}%</strong>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600">•</span>
            Total measurements processed:{' '}
            <strong className="text-gray-900">{analytics.total_measurements.toLocaleString()}</strong>
          </li>
        </ul>
      </div>
    </div>
  );
}

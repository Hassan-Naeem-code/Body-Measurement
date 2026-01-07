'use client';

import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { Analytics } from '@/lib/types';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';

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
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-4 w-64 mt-2" />
        </div>
        <div className="grid md:grid-cols-3 gap-6">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i}>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-8 w-24 mt-2" />
                  </div>
                  <Skeleton className="h-10 w-10 rounded-full" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <Card>
          <CardContent>
            <Skeleton className="h-6 w-40 mb-4" />
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-4 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (analyticsQuery.error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
        {(analyticsQuery.error as Error)?.message}
      </div>
    );
  }

  const analytics = analyticsQuery.data;
  if (!analytics) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded">
        No analytics data available yet
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600 mt-2">Track performance and ROI metrics</p>
      </div>

      {/* Key Metrics */}
      <div className="grid md:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Measurements</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {analytics.total_measurements.toLocaleString()}
              </p>
            </div>
            <div className="text-4xl">📏</div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Average Confidence</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {(analytics.average_confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-4xl">🎯</div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">ROI</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {analytics.revenue_impact.roi_percentage.toFixed(1)}%
              </p>
            </div>
            <div className="text-4xl">💰</div>
          </div>
        </Card>
      </div>

      {/* Size Distribution */}
      <Card>
        <CardContent>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Size Distribution</h2>
        <div className="space-y-3">
          {Object.entries(analytics.size_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([size, count]) => {
              const total = Object.values(analytics.size_distribution).reduce(
                (sum, val) => sum + val,
                0
              );
              const percentage = (count / total) * 100;
              return (
                <div key={size}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-gray-900">Size {size}</span>
                    <span className="text-gray-600">
                      {count} ({percentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-indigo-600 rounded-full h-2"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
        </div>
        </CardContent>
      </Card>

      {/* Popular Products */}
      <Card>
        <CardContent>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Popular Products</h2>
        {analytics.popular_products.length === 0 ? (
          <p className="text-gray-600">No product data available yet</p>
        ) : (
          <div className="space-y-3">
            {analytics.popular_products.map((product, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold">
                    {index + 1}
                  </div>
                  <span className="font-semibold text-gray-900">{product.product_name}</span>
                </div>
                <span className="text-gray-600">{product.measurement_count} measurements</span>
              </div>
            ))}
          </div>
        )}
        </CardContent>
      </Card>

      {/* Revenue Impact */}
      <div className="bg-gradient-to-r from-purple-500 to-pink-600 p-6 rounded-lg text-white">
        <h2 className="text-2xl font-bold mb-6">Revenue Impact</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <p className="text-purple-100">Estimated Conversions</p>
            <p className="text-4xl font-bold mt-2">
              {analytics.revenue_impact.estimated_conversions}
            </p>
            <p className="text-purple-100 mt-1 text-sm">
              Additional sales from accurate sizing
            </p>
          </div>
          <div>
            <p className="text-purple-100">Returns Prevented</p>
            <p className="text-4xl font-bold mt-2">
              {analytics.revenue_impact.estimated_returns_prevented}
            </p>
            <p className="text-purple-100 mt-1 text-sm">
              Fewer returns due to wrong size
            </p>
          </div>
          <div>
            <p className="text-purple-100">ROI Percentage</p>
            <p className="text-4xl font-bold mt-2">
              {analytics.revenue_impact.roi_percentage.toFixed(1)}%
            </p>
            <p className="text-purple-100 mt-1 text-sm">
              Return on investment
            </p>
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">Key Insights</h3>
        <ul className="space-y-2 text-blue-800">
          <li>
            ✓ Most customers fit into size{' '}
            <strong>
              {Object.entries(analytics.size_distribution).sort(([, a], [, b]) => b - a)[0]?.[0] ||
                'N/A'}
            </strong>
          </li>
          <li>
            ✓ Average measurement confidence score:{' '}
            <strong>{(analytics.average_confidence * 100).toFixed(1)}%</strong>
          </li>
          <li>
            ✓ Total measurements processed:{' '}
            <strong>{analytics.total_measurements.toLocaleString()}</strong>
          </li>
          {analytics.popular_products.length > 0 && (
            <li>
              ✓ Most popular product:{' '}
              <strong>{analytics.popular_products[0].product_name}</strong>
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

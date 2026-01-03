'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { UsageStats, Brand } from '@/lib/types';

export default function DashboardPage() {
  const [usage, setUsage] = useState<UsageStats | null>(null);
  const [brand, setBrand] = useState<Brand | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const apiKey = authHelpers.getApiKey();
      const brandData = authHelpers.getBrand();

      if (!apiKey || !brandData) {
        setError('Authentication data not found');
        return;
      }

      setBrand(brandData);
      const usageData = await brandAPI.getUsage(apiKey);
      setUsage(usageData);
    } catch (err: unknown) {
      setError('Failed to load dashboard data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">Welcome back, {brand?.name}!</p>
      </div>

      {/* Stats Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Requests</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {usage?.total_requests || 0}
              </p>
            </div>
            <div className="text-4xl">📊</div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Today</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {usage?.requests_today || 0}
              </p>
            </div>
            <div className="text-4xl">📅</div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">This Month</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {usage?.requests_this_month || 0}
              </p>
            </div>
            <div className="text-4xl">📈</div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg Processing</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {usage?.average_processing_time || 0}
                <span className="text-lg text-gray-600">ms</span>
              </p>
            </div>
            <div className="text-4xl">⚡</div>
          </div>
        </div>
      </div>

      {/* Plan Info */}
      <div className="bg-gradient-to-r from-indigo-500 to-purple-600 p-6 rounded-lg text-white">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-indigo-100">Current Plan</p>
            <p className="text-2xl font-bold mt-1 capitalize">
              {brand?.subscription_tier || 'Free'}
            </p>
            <p className="text-indigo-100 mt-2">
              {usage?.requests_this_month || 0} / {usage?.plan_limit || 1000} requests used
            </p>
          </div>
          <div className="text-6xl">🚀</div>
        </div>
        <div className="mt-4 bg-white bg-opacity-20 rounded-full h-2">
          <div
            className="bg-white rounded-full h-2"
            style={{
              width: `${Math.min(
                ((usage?.requests_this_month || 0) / (usage?.plan_limit || 1000)) * 100,
                100
              )}%`,
            }}
          />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <Link
            href="/dashboard/upload"
            className="p-4 border-2 border-indigo-200 rounded-lg hover:border-indigo-400 hover:bg-indigo-50 transition text-center"
          >
            <div className="text-4xl mb-2">📸</div>
            <p className="font-semibold text-gray-900">Upload Image</p>
            <p className="text-sm text-gray-600 mt-1">Process body measurements</p>
          </Link>

          <Link
            href="/dashboard/products"
            className="p-4 border-2 border-green-200 rounded-lg hover:border-green-400 hover:bg-green-50 transition text-center"
          >
            <div className="text-4xl mb-2">👔</div>
            <p className="font-semibold text-gray-900">Manage Products</p>
            <p className="text-sm text-gray-600 mt-1">Add size charts</p>
          </Link>

          <Link
            href="/dashboard/analytics"
            className="p-4 border-2 border-purple-200 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition text-center"
          >
            <div className="text-4xl mb-2">📈</div>
            <p className="font-semibold text-gray-900">View Analytics</p>
            <p className="text-sm text-gray-600 mt-1">Track performance</p>
          </Link>
        </div>
      </div>

      {/* Getting Started */}
      {usage && usage.total_requests === 0 && (
        <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">Getting Started</h3>
          <p className="text-blue-700 mb-4">
            Welcome to Body Measurement API! Here&apos;s how to get started:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-blue-800">
            <li>Upload your first image to test the measurement system</li>
            <li>Add your products with size charts</li>
            <li>Get your API key and integrate with your e-commerce platform</li>
            <li>Track analytics and ROI in real-time</li>
          </ol>
        </div>
      )}
    </div>
  );
}

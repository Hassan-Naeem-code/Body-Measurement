'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { UsageStats, Brand } from '@/lib/types';
import {
  BarChart3,
  Calendar,
  TrendingUp,
  Zap,
  Upload,
  ShoppingBag,
  LineChart,
  ArrowRight,
  Code2,
  Smartphone,
  ExternalLink,
  Copy,
  CheckCircle2,
} from 'lucide-react';

const formatNumber = (num: number): string => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
};

const formatTime = (ms: number): string => {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + 's';
  return Math.round(ms) + 'ms';
};

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
    } catch (err) {
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 bg-gray-200 rounded animate-pulse" />
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-5">
              <div className="h-4 w-24 bg-gray-200 rounded animate-pulse" />
              <div className="h-8 w-16 mt-3 bg-gray-200 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[300px]">
        <div className="text-center p-6 bg-red-50 border border-red-200 rounded-xl max-w-md">
          <p className="text-red-700">{error}</p>
          <button
            onClick={loadData}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  const stats = [
    {
      label: 'Total Requests',
      value: formatNumber(usage?.total_requests || 0),
      icon: BarChart3,
      bgColor: 'bg-indigo-100',
      iconColor: 'text-indigo-600',
    },
    {
      label: 'Today',
      value: formatNumber(usage?.requests_today || 0),
      icon: Calendar,
      bgColor: 'bg-blue-100',
      iconColor: 'text-blue-600',
    },
    {
      label: 'This Month',
      value: formatNumber(usage?.requests_this_month || 0),
      icon: TrendingUp,
      bgColor: 'bg-green-100',
      iconColor: 'text-green-600',
    },
    {
      label: 'Avg Processing',
      value: formatTime(usage?.average_processing_time || 0),
      icon: Zap,
      bgColor: 'bg-amber-100',
      iconColor: 'text-amber-600',
    },
  ];

  const quickActions = [
    {
      href: '/dashboard/upload',
      icon: Upload,
      title: 'Upload Image',
      description: 'Process body measurements',
      bgColor: 'bg-indigo-600',
    },
    {
      href: '/dashboard/products',
      icon: ShoppingBag,
      title: 'Manage Products',
      description: 'Add size charts',
      bgColor: 'bg-green-600',
    },
    {
      href: '/dashboard/analytics',
      icon: LineChart,
      title: 'View Analytics',
      description: 'Track performance',
      bgColor: 'bg-blue-600',
    },
  ];

  const usagePercent = Math.min(((usage?.requests_this_month || 0) / (usage?.plan_limit || 1000)) * 100, 100);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 mt-1">
            Welcome back, <span className="text-gray-900 font-medium">{brand?.name}</span>
          </p>
        </div>
        <Link
          href="/dashboard/upload"
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
        >
          <Upload className="w-4 h-4" />
          New Measurement
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">{stat.label}</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stat.value}</p>
              </div>
              <div className={`w-12 h-12 rounded-xl ${stat.bgColor} flex items-center justify-center`}>
                <stat.icon className={`w-6 h-6 ${stat.iconColor}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Plan Usage - Compact */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-sm font-medium text-gray-500">Monthly Usage</p>
            <p className="text-lg font-semibold text-gray-900">
              {formatNumber(usage?.requests_this_month || 0)} / {formatNumber(usage?.plan_limit || 1000)} requests
            </p>
          </div>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium capitalize">
            {brand?.subscription_tier || 'Free'} Plan
          </span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-600 rounded-full transition-all duration-500"
            style={{ width: `${usagePercent}%` }}
          />
        </div>
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {quickActions.map((action) => (
            <Link
              key={action.href}
              href={action.href}
              className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md hover:border-gray-300 transition-all group"
            >
              <div className={`w-12 h-12 rounded-xl ${action.bgColor} flex items-center justify-center mb-4`}>
                <action.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-gray-900 group-hover:text-indigo-600 transition-colors">
                {action.title}
              </h3>
              <p className="text-sm text-gray-500 mt-1">{action.description}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* SDK Integration Section */}
      <div className="bg-gradient-to-br from-indigo-50 via-white to-purple-50 rounded-2xl border border-indigo-100 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center flex-shrink-0">
              <Code2 className="w-7 h-7 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Multi-Platform SDK Support</h2>
              <p className="text-gray-600 mt-1">
                Integrate body measurements into your apps with our production-ready SDKs
              </p>
            </div>
          </div>
          <Link
            href="/dashboard/sdk"
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white rounded-xl font-medium hover:bg-indigo-700 transition-colors whitespace-nowrap"
          >
            <Smartphone className="w-4 h-4" />
            View SDK Docs
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {/* SDK Platform Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-6">
          {[
            { name: 'React Native', status: 'Production', icon: '⚛️', color: 'bg-blue-500' },
            { name: 'Flutter', status: 'Production', icon: '🐦', color: 'bg-cyan-500' },
            { name: 'iOS Swift', status: 'Production', icon: '🍎', color: 'bg-gray-800' },
            { name: 'Android', status: 'Production', icon: '🤖', color: 'bg-green-500' },
            { name: 'Web SDK', status: 'Production', icon: '🌐', color: 'bg-indigo-500' },
            { name: 'REST API', status: 'Production', icon: '🔌', color: 'bg-purple-500' },
          ].map((sdk) => (
            <div
              key={sdk.name}
              className="bg-white rounded-xl border border-gray-200 p-4 hover:shadow-md hover:border-indigo-200 transition-all"
            >
              <div className="text-2xl mb-2">{sdk.icon}</div>
              <h3 className="font-semibold text-gray-900 text-sm">{sdk.name}</h3>
              <span className="inline-flex items-center gap-1 mt-1 text-xs text-green-600 font-medium">
                <CheckCircle2 className="w-3 h-3" />
                {sdk.status}
              </span>
            </div>
          ))}
        </div>

        {/* Quick Install Snippet */}
        <div className="mt-6 bg-gray-900 rounded-xl p-4 overflow-x-auto">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400 font-medium">Quick Install</span>
            <div className="flex gap-2">
              <span className="px-2 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded">npm</span>
              <span className="px-2 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 rounded">pub</span>
            </div>
          </div>
          <code className="text-sm text-green-400 font-mono">
            npm install @body-measurement/react-native-sdk
          </code>
        </div>
      </div>
    </div>
  );
}

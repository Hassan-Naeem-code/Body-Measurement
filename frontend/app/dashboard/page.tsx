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
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Rocket,
} from 'lucide-react';

// Utility function to format numbers
const formatNumber = (num: number, decimals: number = 0): string => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toFixed(decimals);
};

// Utility function to format processing time
const formatProcessingTime = (ms: number): string => {
  if (ms >= 1000) {
    return (ms / 1000).toFixed(2) + 's';
  }
  return Math.round(ms) + 'ms';
};

// Loading skeleton component
const StatCardSkeleton = () => (
  <div className="stat-card animate-pulse">
    <div className="flex items-start justify-between">
      <div className="space-y-3">
        <div className="h-4 w-24 bg-muted rounded skeleton" />
        <div className="h-8 w-16 bg-muted rounded skeleton" />
      </div>
      <div className="h-12 w-12 bg-muted rounded-xl skeleton" />
    </div>
  </div>
);

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
      <div className="space-y-6 animate-fade-in">
        {/* Header Skeleton */}
        <div className="space-y-2">
          <div className="h-8 w-48 bg-muted rounded skeleton" />
          <div className="h-5 w-64 bg-muted rounded skeleton" />
        </div>

        {/* Stats Grid Skeleton */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 lg:gap-6">
          {[...Array(4)].map((_, i) => (
            <StatCardSkeleton key={i} />
          ))}
        </div>

        {/* Plan Skeleton */}
        <div className="h-40 bg-muted rounded-2xl skeleton" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center space-y-4 p-8 rounded-2xl bg-destructive-muted border border-destructive/20 max-w-md">
          <div className="w-16 h-16 mx-auto rounded-full bg-destructive/10 flex items-center justify-center">
            <span className="text-destructive text-2xl">!</span>
          </div>
          <h3 className="text-lg font-semibold text-foreground">Something went wrong</h3>
          <p className="text-muted-foreground">{error}</p>
          <button
            onClick={loadData}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary-hover transition-colors"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  const usagePercentage = Math.min(
    ((usage?.requests_this_month || 0) / (usage?.plan_limit || 1000)) * 100,
    100
  );

  const stats = [
    {
      label: 'Total Requests',
      value: formatNumber(usage?.total_requests || 0),
      icon: BarChart3,
      color: 'primary',
      bgColor: 'bg-primary-muted',
      iconColor: 'text-primary',
    },
    {
      label: 'Today',
      value: formatNumber(usage?.requests_today || 0),
      icon: Calendar,
      color: 'info',
      bgColor: 'bg-info-muted',
      iconColor: 'text-info',
    },
    {
      label: 'This Month',
      value: formatNumber(usage?.requests_this_month || 0),
      icon: TrendingUp,
      color: 'success',
      bgColor: 'bg-success-muted',
      iconColor: 'text-success',
    },
    {
      label: 'Avg Processing',
      value: formatProcessingTime(usage?.average_processing_time || 0),
      icon: Zap,
      color: 'warning',
      bgColor: 'bg-warning-muted',
      iconColor: 'text-warning',
    },
  ];

  const quickActions = [
    {
      href: '/dashboard/upload',
      icon: Upload,
      title: 'Upload Image',
      description: 'Process body measurements',
      gradient: 'from-primary to-primary-hover',
    },
    {
      href: '/dashboard/products',
      icon: ShoppingBag,
      title: 'Manage Products',
      description: 'Add size charts',
      gradient: 'from-success to-emerald-600',
    },
    {
      href: '/dashboard/analytics',
      icon: LineChart,
      title: 'View Analytics',
      description: 'Track performance',
      gradient: 'from-info to-cyan-600',
    },
  ];

  return (
    <div className="space-y-6 lg:space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl lg:text-3xl font-bold tracking-tight text-foreground">
            Dashboard
          </h1>
          <p className="text-muted-foreground">
            Welcome back, <span className="text-foreground font-medium">{brand?.name}</span>
          </p>
        </div>
        <Link
          href="/dashboard/upload"
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary-hover active:bg-primary-active transition-all duration-200 shadow-sm hover:shadow-md group"
        >
          <Upload className="w-4 h-4" />
          <span>New Measurement</span>
          <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 lg:gap-6">
        {stats.map((stat, index) => (
          <div
            key={stat.label}
            className="stat-card group"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                <p className="text-3xl lg:text-4xl font-bold tracking-tight text-foreground">
                  {stat.value}
                </p>
              </div>
              <div
                className={`icon-container icon-container-md ${stat.bgColor} transition-transform duration-200 group-hover:scale-110`}
              >
                <stat.icon className={`w-6 h-6 ${stat.iconColor}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Plan Info Card */}
      <div className="relative overflow-hidden rounded-2xl gradient-primary p-6 lg:p-8 text-white">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-white rounded-full blur-3xl transform -translate-x-1/2 translate-y-1/2" />
        </div>

        <div className="relative flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-4 flex-1">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-sm font-medium backdrop-blur-sm">
              <Sparkles className="w-4 h-4" />
              <span>Current Plan</span>
            </div>
            <div>
              <h2 className="text-3xl lg:text-4xl font-bold capitalize">
                {brand?.subscription_tier || 'Free'}
              </h2>
              <p className="text-white/80 mt-2">
                {formatNumber(usage?.requests_this_month || 0)} / {formatNumber(usage?.plan_limit || 1000)} requests used this month
              </p>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2 max-w-md">
              <div className="flex justify-between text-sm">
                <span className="text-white/80">Usage</span>
                <span className="font-medium">{usagePercentage.toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-white/20 rounded-full overflow-hidden backdrop-blur-sm">
                <div
                  className="h-full bg-white rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${usagePercentage}%` }}
                />
              </div>
            </div>
          </div>

          <div className="hidden lg:flex items-center justify-center">
            <div className="w-24 h-24 rounded-2xl bg-white/10 backdrop-blur-sm flex items-center justify-center">
              <Rocket className="w-12 h-12 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-foreground">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {quickActions.map((action, index) => (
            <Link
              key={action.href}
              href={action.href}
              className="action-card group"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div
                className={`w-14 h-14 rounded-xl bg-gradient-to-br ${action.gradient} flex items-center justify-center mb-4 transition-transform duration-200 group-hover:scale-110 shadow-lg`}
              >
                <action.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
                {action.title}
              </h3>
              <p className="text-sm text-muted-foreground mt-1">{action.description}</p>
              <ArrowRight className="w-5 h-5 text-muted-foreground mt-3 transition-all duration-200 opacity-0 group-hover:opacity-100 group-hover:translate-x-1" />
            </Link>
          ))}
        </div>
      </div>

      {/* Getting Started */}
      {usage && usage.total_requests === 0 && (
        <div className="rounded-2xl border-2 border-info/20 bg-info-muted/50 p-6 lg:p-8 animate-slide-up">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-info/10 flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-info" />
            </div>
            <div className="flex-1 space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-foreground">Getting Started</h3>
                <p className="text-muted-foreground mt-1">
                  Welcome to Body Measurement API! Here&apos;s how to get started:
                </p>
              </div>
              <ol className="space-y-3">
                {[
                  'Upload your first image to test the measurement system',
                  'Add your products with size charts',
                  'Get your API key and integrate with your e-commerce platform',
                  'Track analytics and ROI in real-time',
                ].map((step, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-info/10 flex items-center justify-center text-xs font-bold text-info">
                      {index + 1}
                    </div>
                    <span className="text-foreground">{step}</span>
                  </li>
                ))}
              </ol>
              <div className="pt-2">
                <Link
                  href="/dashboard/upload"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-info text-white rounded-lg font-medium hover:bg-info/90 transition-colors"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  Start First Measurement
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

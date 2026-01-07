'use client';

import { useState, useEffect } from 'react';
import { authHelpers } from '@/lib/auth';
import type { Brand } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

export default function ApiKeysPage() {
  const [brand, setBrand] = useState<Brand | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const brandData = authHelpers.getBrand();
    setBrand(brandData);
  }, []);

  const handleCopy = () => {
    if (brand?.api_key) {
      navigator.clipboard.writeText(brand.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!brand) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">API Keys</h1>
        <p className="text-gray-600 mt-2">Manage your API credentials</p>
      </div>

      {/* API Key Display */}
      <Card>
        <CardContent>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Your API Key</h2>
        <p className="text-gray-600 mb-4">
          Use this key to authenticate API requests from your application
        </p>

        <div className="bg-gray-50 p-4 rounded-lg border border-gray-300">
          <div className="flex items-center justify-between">
            <code className="text-sm font-mono text-gray-900 break-all">
              {brand.api_key}
            </code>
            <Button onClick={handleCopy} className="ml-4 whitespace-nowrap">
              {copied ? '✓ Copied!' : 'Copy'}
            </Button>
          </div>
        </div>

        <div className="mt-4 bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
          <p className="text-sm text-yellow-800">
            <strong>⚠️ Security Warning:</strong> Keep your API key secret. Don&apos;t share it publicly
            or commit it to version control.
          </p>
        </div>
        </CardContent>
      </Card>

      {/* API Documentation */}
      <Card>
        <CardContent>
        <h2 className="text-xl font-bold text-gray-900 mb-4">API Usage</h2>
        <p className="text-gray-600 mb-4">
          Here are some example API calls to get you started:
        </p>

        <div className="space-y-4">
          {/* Example 1: Process Image */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Process Image</h3>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
                <code>{`curl -X POST "http://localhost:8000/api/v1/measurements/process?api_key=${brand.api_key}" \\
  -F "file=@image.jpg"`}</code>
              </pre>
            </div>
          </div>

          {/* Example 2: Get Profile */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Get Profile</h3>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
                <code>{`curl "http://localhost:8000/api/v1/brands/me?api_key=${brand.api_key}"`}</code>
              </pre>
            </div>
          </div>

          {/* Example 3: Add Product */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Add Product</h3>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
                <code>{`curl -X POST "http://localhost:8000/api/v1/brands/products?api_key=${brand.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Classic T-Shirt",
    "category": "T-Shirts",
    "size_chart": {
      "S": {"chest": 90, "waist": 75},
      "M": {"chest": 95, "waist": 80},
      "L": {"chest": 100, "waist": 85}
    }
  }'`}</code>
              </pre>
            </div>
          </div>
        </div>
        </CardContent>
      </Card>

      {/* Account Info */}
      <Card>
        <CardContent>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Account Information</h2>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-600">Brand Name:</span>
            <span className="font-semibold text-gray-900">{brand.name}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Email:</span>
            <span className="font-semibold text-gray-900">{brand.email}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Subscription:</span>
            <span className="font-semibold text-gray-900 capitalize">
              {brand.subscription_tier}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Status:</span>
            <span
              className={`font-semibold ${
                brand.is_active ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {brand.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Member Since:</span>
            <span className="font-semibold text-gray-900">
              {new Date(brand.created_at).toLocaleDateString()}
            </span>
          </div>
        </div>
        </CardContent>
      </Card>

      {/* API Documentation Link */}
      <div className="bg-indigo-50 border border-indigo-200 p-6 rounded-lg text-center">
        <h3 className="text-lg font-semibold text-indigo-900 mb-2">
          Need More Help?
        </h3>
        <p className="text-indigo-700 mb-4">
          Check out the full API documentation for detailed information
        </p>
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center rounded-md bg-primary text-primary-foreground px-6 py-3 font-medium hover:bg-primary/90"
        >
          View API Docs
        </a>
      </div>
    </div>
  );
}

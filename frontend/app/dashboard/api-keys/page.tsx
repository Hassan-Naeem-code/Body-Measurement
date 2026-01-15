'use client';

import { useState, useEffect } from 'react';
import { authHelpers } from '@/lib/auth';
import type { Brand } from '@/lib/types';
import { Key, Copy, Check, Shield, Code, User, ExternalLink, Terminal } from 'lucide-react';

export default function ApiKeysPage() {
  const [brand, setBrand] = useState<Brand | null>(null);
  const [copied, setCopied] = useState(false);
  const [copiedExample, setCopiedExample] = useState<string | null>(null);

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

  const copyExample = (code: string, name: string) => {
    navigator.clipboard.writeText(code);
    setCopiedExample(name);
    setTimeout(() => setCopiedExample(null), 2000);
  };

  if (!brand) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const examples = [
    {
      name: 'Process Image',
      description: 'Upload and process a body image',
      code: `curl -X POST "http://localhost:8000/api/v1/measurements/process?api_key=${brand.api_key}" -F "file=@image.jpg"`,
    },
    {
      name: 'Get Profile',
      description: 'Retrieve your account information',
      code: `curl "http://localhost:8000/api/v1/brands/me?api_key=${brand.api_key}"`,
    },
    {
      name: 'Add Product',
      description: 'Create a new product with size chart',
      code: `curl -X POST "http://localhost:8000/api/v1/brands/products?api_key=${brand.api_key}" -H "Content-Type: application/json" -d '{"name": "T-Shirt", "category": "Tops", "size_chart": {"S": {"chest": 90}, "M": {"chest": 95}}}'`,
    },
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">API Keys</h1>
        <p className="text-gray-500 mt-1">Manage your API credentials</p>
      </div>

      {/* API Key Card */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-5 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
              <Key className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Your API Key</h2>
              <p className="text-sm text-gray-500">Use this key to authenticate API requests</p>
            </div>
          </div>
        </div>
        <div className="p-5">
          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <code className="flex-1 text-sm font-mono text-gray-800 break-all">
              {brand.api_key}
            </code>
            <button
              onClick={handleCopy}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                copied
                  ? 'bg-green-100 text-green-700'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <div className="mt-4 flex items-start gap-3 p-3 bg-amber-50 rounded-lg border border-amber-200">
            <Shield className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-amber-800">
              <strong>Security:</strong> Keep your API key secret. Never share it publicly or commit to version control.
            </p>
          </div>
        </div>
      </div>

      {/* API Examples */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-5 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center">
              <Terminal className="w-5 h-5 text-gray-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Quick Start Examples</h2>
              <p className="text-sm text-gray-500">Copy and run these commands to test the API</p>
            </div>
          </div>
        </div>
        <div className="divide-y divide-gray-100">
          {examples.map((example) => (
            <div key={example.name} className="p-5">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="font-medium text-gray-900">{example.name}</h3>
                  <p className="text-sm text-gray-500">{example.description}</p>
                </div>
                <button
                  onClick={() => copyExample(example.code, example.name)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    copiedExample === example.name
                      ? 'bg-green-100 text-green-700'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {copiedExample === example.name ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedExample === example.name ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                <code className="text-sm text-gray-100 whitespace-pre-wrap break-all font-mono">
                  {example.code}
                </code>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Account Info */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-5 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
              <User className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Account Information</h2>
              <p className="text-sm text-gray-500">Your account details</p>
            </div>
          </div>
        </div>
        <div className="p-5">
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Brand Name</p>
              <p className="text-sm font-semibold text-gray-900 mt-1">{brand.name}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Email</p>
              <p className="text-sm font-semibold text-gray-900 mt-1">{brand.email}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Plan</p>
              <p className="text-sm font-semibold text-gray-900 mt-1 capitalize">{brand.subscription_tier}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Status</p>
              <p className={`text-sm font-semibold mt-1 ${brand.is_active ? 'text-green-600' : 'text-red-600'}`}>
                {brand.is_active ? 'Active' : 'Inactive'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* API Docs Link */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-lg">Full API Documentation</h3>
            <p className="text-indigo-100 text-sm mt-1">Explore all endpoints and parameters</p>
          </div>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 bg-white text-indigo-600 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            View Docs
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
}

'use client';

import { useState, useEffect } from 'react';
import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { Product, ProductRequest } from '@/lib/types';

export default function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState<ProductRequest>({
    name: '',
    category: '',
    size_chart: {
      XS: { chest: 0, waist: 0, hip: 0, inseam: 0 },
      S: { chest: 0, waist: 0, hip: 0, inseam: 0 },
      M: { chest: 0, waist: 0, hip: 0, inseam: 0 },
      L: { chest: 0, waist: 0, hip: 0, inseam: 0 },
      XL: { chest: 0, waist: 0, hip: 0, inseam: 0 },
    },
  });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadProducts();
  }, []);

  const loadProducts = async () => {
    try {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) {
        setError('API key not found');
        return;
      }

      const data = await brandAPI.getProducts(apiKey);
      setProducts(data);
    } catch (err: unknown) {
      setError('Failed to load products');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError('');

    try {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) {
        setError('API key not found');
        return;
      }

      await brandAPI.addProduct(apiKey, formData);
      setShowForm(false);
      loadProducts();

      // Reset form
      setFormData({
        name: '',
        category: '',
        size_chart: {
          XS: { chest: 0, waist: 0, hip: 0, inseam: 0 },
          S: { chest: 0, waist: 0, hip: 0, inseam: 0 },
          M: { chest: 0, waist: 0, hip: 0, inseam: 0 },
          L: { chest: 0, waist: 0, hip: 0, inseam: 0 },
          XL: { chest: 0, waist: 0, hip: 0, inseam: 0 },
        },
      });
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to add product');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSizeChange = (size: string, field: string, value: string) => {
    setFormData({
      ...formData,
      size_chart: {
        ...formData.size_chart,
        [size]: {
          ...formData.size_chart[size],
          [field]: parseFloat(value) || 0,
        },
      },
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading products...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Products</h1>
          <p className="text-gray-600 mt-2">Manage your product catalog and size charts</p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
        >
          {showForm ? 'Cancel' : '+ Add Product'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Add Product Form */}
      {showForm && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Add New Product</h2>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Product Name
                </label>
                <input
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="e.g., Classic T-Shirt"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Category
                </label>
                <input
                  type="text"
                  required
                  value={formData.category}
                  onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="e.g., T-Shirts, Jeans, Jackets"
                />
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Size Chart (cm)</h3>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="border border-gray-300 px-4 py-2 text-left">Size</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">Chest</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">Waist</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">Hip</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">Inseam</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['XS', 'S', 'M', 'L', 'XL'].map((size) => (
                      <tr key={size}>
                        <td className="border border-gray-300 px-4 py-2 font-semibold">{size}</td>
                        <td className="border border-gray-300 px-2 py-2">
                          <input
                            type="number"
                            step="0.1"
                            value={formData.size_chart[size]?.chest || ''}
                            onChange={(e) => handleSizeChange(size, 'chest', e.target.value)}
                            className="w-full px-2 py-1 border border-gray-300 rounded"
                          />
                        </td>
                        <td className="border border-gray-300 px-2 py-2">
                          <input
                            type="number"
                            step="0.1"
                            value={formData.size_chart[size]?.waist || ''}
                            onChange={(e) => handleSizeChange(size, 'waist', e.target.value)}
                            className="w-full px-2 py-1 border border-gray-300 rounded"
                          />
                        </td>
                        <td className="border border-gray-300 px-2 py-2">
                          <input
                            type="number"
                            step="0.1"
                            value={formData.size_chart[size]?.hip || ''}
                            onChange={(e) => handleSizeChange(size, 'hip', e.target.value)}
                            className="w-full px-2 py-1 border border-gray-300 rounded"
                          />
                        </td>
                        <td className="border border-gray-300 px-2 py-2">
                          <input
                            type="number"
                            step="0.1"
                            value={formData.size_chart[size]?.inseam || ''}
                            onChange={(e) => handleSizeChange(size, 'inseam', e.target.value)}
                            className="w-full px-2 py-1 border border-gray-300 rounded"
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition disabled:bg-indigo-400 disabled:cursor-not-allowed"
            >
              {submitting ? 'Adding Product...' : 'Add Product'}
            </button>
          </form>
        </div>
      )}

      {/* Products List */}
      <div className="space-y-4">
        {products.length === 0 ? (
          <div className="bg-white p-12 rounded-lg shadow-sm border border-gray-200 text-center">
            <div className="text-6xl mb-4">👔</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No products yet</h3>
            <p className="text-gray-600 mb-4">
              Add your first product with size chart to start making size recommendations
            </p>
            <button
              onClick={() => setShowForm(true)}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
            >
              Add Your First Product
            </button>
          </div>
        ) : (
          products.map((product) => (
            <div
              key={product.id}
              className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-xl font-bold text-gray-900">{product.name}</h3>
                  <p className="text-gray-600">{product.category}</p>
                  <p className="text-sm text-gray-500 mt-1">
                    Added {new Date(product.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Size Chart (cm)</h4>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="border border-gray-300 px-3 py-2 text-left">Size</th>
                      <th className="border border-gray-300 px-3 py-2 text-left">Chest</th>
                      <th className="border border-gray-300 px-3 py-2 text-left">Waist</th>
                      <th className="border border-gray-300 px-3 py-2 text-left">Hip</th>
                      <th className="border border-gray-300 px-3 py-2 text-left">Inseam</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(product.size_chart).map(([size, measurements]) => (
                      <tr key={size}>
                        <td className="border border-gray-300 px-3 py-2 font-semibold">{size}</td>
                        <td className="border border-gray-300 px-3 py-2">
                          {measurements.chest || '-'}
                        </td>
                        <td className="border border-gray-300 px-3 py-2">
                          {measurements.waist || '-'}
                        </td>
                        <td className="border border-gray-300 px-3 py-2">
                          {measurements.hip || '-'}
                        </td>
                        <td className="border border-gray-300 px-3 py-2">
                          {measurements.inseam || '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { productsAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import { toast } from 'sonner';
import { useConfirmDialog } from '@/components/ui/confirm-dialog';
import type { ProductWithSizeCharts, SizeChartCreate } from '@/lib/types';
import {
  Package,
  Plus,
  Pencil,
  Trash2,
  ChevronDown,
  ChevronUp,
  Ruler,
  Tag,
  Users,
  Check,
  X,
} from 'lucide-react';

const CATEGORIES = ['tops', 'bottoms', 'dresses', 'outerwear', 'activewear', 'swimwear'];
const SIZES = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL'];
const FIT_TYPES = ['tight', 'regular', 'loose'] as const;

export default function ProductsPage() {
  const queryClient = useQueryClient();
  const { confirm } = useConfirmDialog();
  const [showAddProduct, setShowAddProduct] = useState(false);
  const [expandedProduct, setExpandedProduct] = useState<string | null>(null);
  const [editingSizeChart, setEditingSizeChart] = useState<string | null>(null);
  const [addingSizeToProduct, setAddingSizeToProduct] = useState<string | null>(null);

  // Form states
  const [productForm, setProductForm] = useState({
    name: '',
    sku: '',
    category: 'tops',
    subcategory: '',
    gender: 'unisex' as 'male' | 'female' | 'unisex',
    description: '',
  });

  const [sizeChartForm, setSizeChartForm] = useState<SizeChartCreate>({
    size_name: 'M',
    chest_min: undefined,
    chest_max: undefined,
    waist_min: undefined,
    waist_max: undefined,
    hip_min: undefined,
    hip_max: undefined,
    height_min: undefined,
    height_max: undefined,
    fit_type: 'regular',
    display_order: 0,
  });

  // Queries
  const productsQuery = useQuery({
    queryKey: ['products-enhanced'],
    queryFn: async () => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return productsAPI.getProducts(apiKey);
    },
  });

  // Mutations
  const createProductMutation = useMutation({
    mutationFn: async () => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return productsAPI.createProduct(apiKey, productForm);
    },
    onSuccess: () => {
      toast.success('Product created successfully');
      setShowAddProduct(false);
      setProductForm({
        name: '',
        sku: '',
        category: 'tops',
        subcategory: '',
        gender: 'unisex',
        description: '',
      });
      queryClient.invalidateQueries({ queryKey: ['products-enhanced'] });
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to create product');
    },
  });

  const deleteProductMutation = useMutation({
    mutationFn: async (productId: string) => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return productsAPI.deleteProduct(apiKey, productId);
    },
    onSuccess: () => {
      toast.success('Product deleted');
      queryClient.invalidateQueries({ queryKey: ['products-enhanced'] });
    },
  });

  const addSizeChartMutation = useMutation({
    mutationFn: async ({ productId, data }: { productId: string; data: SizeChartCreate }) => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return productsAPI.addSizeChart(apiKey, productId, data);
    },
    onSuccess: () => {
      toast.success('Size chart added');
      setAddingSizeToProduct(null);
      resetSizeChartForm();
      queryClient.invalidateQueries({ queryKey: ['products-enhanced'] });
    },
  });

  const deleteSizeChartMutation = useMutation({
    mutationFn: async (chartId: string) => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return productsAPI.deleteSizeChart(apiKey, chartId);
    },
    onSuccess: () => {
      toast.success('Size chart deleted');
      queryClient.invalidateQueries({ queryKey: ['products-enhanced'] });
    },
  });

  const resetSizeChartForm = () => {
    setSizeChartForm({
      size_name: 'M',
      chest_min: undefined,
      chest_max: undefined,
      waist_min: undefined,
      waist_max: undefined,
      hip_min: undefined,
      hip_max: undefined,
      height_min: undefined,
      height_max: undefined,
      fit_type: 'regular',
      display_order: 0,
    });
  };

  const products = productsQuery.data?.products || [];

  if (productsQuery.isLoading) {
    return (
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="h-8 w-40 bg-gray-200 rounded animate-pulse" />
            <div className="h-4 w-64 mt-2 bg-gray-200 rounded animate-pulse" />
          </div>
          <div className="h-10 w-32 bg-gray-200 rounded animate-pulse" />
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="h-6 w-48 bg-gray-200 rounded animate-pulse" />
              <div className="h-4 w-32 mt-2 bg-gray-200 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Products & Size Charts</h1>
          <p className="text-gray-500 mt-1">Manage your product catalog and size mappings</p>
        </div>
        <button
          onClick={() => setShowAddProduct(true)}
          className="flex items-center gap-2 px-4 py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Product
        </button>
      </div>

      {/* Add Product Modal */}
      {showAddProduct && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Add New Product</h2>
              <p className="text-sm text-gray-500 mt-1">Create a product to add size charts</p>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                createProductMutation.mutate();
              }}
              className="p-6 space-y-4"
            >
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Product Name</label>
                  <input
                    type="text"
                    required
                    value={productForm.name}
                    onChange={(e) => setProductForm({ ...productForm, name: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Classic Fit T-Shirt"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">SKU (Optional)</label>
                  <input
                    type="text"
                    value={productForm.sku}
                    onChange={(e) => setProductForm({ ...productForm, sku: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="TSH-001"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                  <select
                    value={productForm.category}
                    onChange={(e) => setProductForm({ ...productForm, category: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    {CATEGORIES.map((cat) => (
                      <option key={cat} value={cat}>
                        {cat.charAt(0).toUpperCase() + cat.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                  <select
                    value={productForm.gender}
                    onChange={(e) => setProductForm({ ...productForm, gender: e.target.value as 'male' | 'female' | 'unisex' })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="unisex">Unisex</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Subcategory</label>
                  <input
                    type="text"
                    value={productForm.subcategory}
                    onChange={(e) => setProductForm({ ...productForm, subcategory: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="T-Shirt, Jeans, etc."
                  />
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                  <textarea
                    value={productForm.description}
                    onChange={(e) => setProductForm({ ...productForm, description: e.target.value })}
                    rows={2}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Product description..."
                  />
                </div>
              </div>
              <div className="flex justify-end gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowAddProduct(false)}
                  className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={createProductMutation.isPending}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                >
                  {createProductMutation.isPending ? 'Creating...' : 'Create Product'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Products List */}
      {products.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Package className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No products yet</h3>
          <p className="text-gray-500 mb-6">Add your first product to start creating size charts</p>
          <button
            onClick={() => setShowAddProduct(true)}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Add Your First Product
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {products.map((product) => (
            <div key={product.id} className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              {/* Product Header */}
              <div
                className="p-5 flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => setExpandedProduct(expandedProduct === product.id ? null : product.id)}
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center">
                    <Package className="w-6 h-6 text-indigo-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">{product.name}</h3>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">
                        <Tag className="w-3 h-3" />
                        {product.category}
                      </span>
                      {product.gender && (
                        <span className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-blue-50 text-blue-600 rounded-full">
                          <Users className="w-3 h-3" />
                          {product.gender}
                        </span>
                      )}
                      <span className="text-xs text-gray-500">
                        {product.size_charts?.length || 0} sizes
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={async (e) => {
                      e.stopPropagation();
                      const confirmed = await confirm({
                        title: 'Delete Product',
                        message: `Are you sure you want to delete "${product.name}"? This will also delete all associated size charts.`,
                        confirmText: 'Delete',
                        cancelText: 'Cancel',
                        variant: 'danger',
                      });
                      if (confirmed) {
                        deleteProductMutation.mutate(product.id);
                      }
                    }}
                    className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  {expandedProduct === product.id ? (
                    <ChevronUp className="w-5 h-5 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                  )}
                </div>
              </div>

              {/* Expanded Size Charts */}
              {expandedProduct === product.id && (
                <div className="border-t border-gray-200 p-5 bg-gray-50">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-medium text-gray-900 flex items-center gap-2">
                      <Ruler className="w-4 h-4 text-gray-500" />
                      Size Charts
                    </h4>
                    <button
                      onClick={() => setAddingSizeToProduct(product.id)}
                      className="flex items-center gap-1 text-sm px-3 py-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      <Plus className="w-4 h-4" />
                      Add Size
                    </button>
                  </div>

                  {/* Add Size Chart Form */}
                  {addingSizeToProduct === product.id && (
                    <div className="bg-white rounded-lg border border-gray-200 p-4 mb-4">
                      <h5 className="font-medium text-gray-900 mb-3">Add New Size</h5>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Size Name</label>
                          <select
                            value={sizeChartForm.size_name}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, size_name: e.target.value })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                          >
                            {SIZES.map((s) => (
                              <option key={s} value={s}>{s}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Fit Type</label>
                          <select
                            value={sizeChartForm.fit_type}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, fit_type: e.target.value as 'tight' | 'regular' | 'loose' })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                          >
                            {FIT_TYPES.map((f) => (
                              <option key={f} value={f}>{f.charAt(0).toUpperCase() + f.slice(1)}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Chest Min (cm)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={sizeChartForm.chest_min || ''}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, chest_min: e.target.value ? parseFloat(e.target.value) : undefined })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                            placeholder="86"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Chest Max (cm)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={sizeChartForm.chest_max || ''}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, chest_max: e.target.value ? parseFloat(e.target.value) : undefined })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                            placeholder="92"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Waist Min (cm)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={sizeChartForm.waist_min || ''}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, waist_min: e.target.value ? parseFloat(e.target.value) : undefined })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                            placeholder="71"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Waist Max (cm)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={sizeChartForm.waist_max || ''}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, waist_max: e.target.value ? parseFloat(e.target.value) : undefined })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                            placeholder="78"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Hip Min (cm)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={sizeChartForm.hip_min || ''}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, hip_min: e.target.value ? parseFloat(e.target.value) : undefined })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                            placeholder="91"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Hip Max (cm)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={sizeChartForm.hip_max || ''}
                            onChange={(e) => setSizeChartForm({ ...sizeChartForm, hip_max: e.target.value ? parseFloat(e.target.value) : undefined })}
                            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-lg"
                            placeholder="97"
                          />
                        </div>
                      </div>
                      <div className="flex justify-end gap-2 mt-4">
                        <button
                          onClick={() => {
                            setAddingSizeToProduct(null);
                            resetSizeChartForm();
                          }}
                          className="px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={() => addSizeChartMutation.mutate({ productId: product.id, data: sizeChartForm })}
                          disabled={addSizeChartMutation.isPending}
                          className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                        >
                          {addSizeChartMutation.isPending ? 'Adding...' : 'Add Size'}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Size Charts Table */}
                  {product.size_charts && product.size_charts.length > 0 ? (
                    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-gray-50 border-b border-gray-200">
                            <tr>
                              <th className="px-4 py-3 text-left font-medium text-gray-600">Size</th>
                              <th className="px-4 py-3 text-left font-medium text-gray-600">Fit</th>
                              <th className="px-4 py-3 text-left font-medium text-gray-600">Chest (cm)</th>
                              <th className="px-4 py-3 text-left font-medium text-gray-600">Waist (cm)</th>
                              <th className="px-4 py-3 text-left font-medium text-gray-600">Hip (cm)</th>
                              <th className="px-4 py-3 text-right font-medium text-gray-600">Actions</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-100">
                            {product.size_charts.map((chart) => (
                              <tr key={chart.id} className="hover:bg-gray-50">
                                <td className="px-4 py-3">
                                  <span className="font-semibold text-gray-900">{chart.size_name}</span>
                                </td>
                                <td className="px-4 py-3">
                                  <span className={`inline-flex px-2 py-0.5 text-xs font-medium rounded-full ${
                                    chart.fit_type === 'tight' ? 'bg-orange-100 text-orange-700' :
                                    chart.fit_type === 'loose' ? 'bg-blue-100 text-blue-700' :
                                    'bg-gray-100 text-gray-700'
                                  }`}>
                                    {chart.fit_type}
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-gray-600">
                                  {chart.chest_min && chart.chest_max ? `${chart.chest_min} - ${chart.chest_max}` : '-'}
                                </td>
                                <td className="px-4 py-3 text-gray-600">
                                  {chart.waist_min && chart.waist_max ? `${chart.waist_min} - ${chart.waist_max}` : '-'}
                                </td>
                                <td className="px-4 py-3 text-gray-600">
                                  {chart.hip_min && chart.hip_max ? `${chart.hip_min} - ${chart.hip_max}` : '-'}
                                </td>
                                <td className="px-4 py-3 text-right">
                                  <button
                                    onClick={async () => {
                                      const confirmed = await confirm({
                                        title: 'Delete Size Chart',
                                        message: `Are you sure you want to delete the size "${chart.size_name}" chart?`,
                                        confirmText: 'Delete',
                                        cancelText: 'Cancel',
                                        variant: 'danger',
                                      });
                                      if (confirmed) {
                                        deleteSizeChartMutation.mutate(chart.id);
                                      }
                                    }}
                                    className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                                  >
                                    <Trash2 className="w-4 h-4" />
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
                      <Ruler className="w-8 h-8 text-gray-300 mx-auto mb-2" />
                      <p className="text-gray-500 text-sm">No size charts yet. Add sizes to enable recommendations.</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Info Card */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-6">
        <h3 className="font-semibold text-gray-900 mb-2">How Size Recommendations Work</h3>
        <ul className="space-y-2 text-sm text-gray-600">
          <li className="flex items-start gap-2">
            <Check className="w-4 h-4 text-indigo-600 mt-0.5 flex-shrink-0" />
            <span>Define min/max measurement ranges for each size (chest, waist, hip)</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-4 h-4 text-indigo-600 mt-0.5 flex-shrink-0" />
            <span>Our AI compares customer measurements against your size charts</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-4 h-4 text-indigo-600 mt-0.5 flex-shrink-0" />
            <span>Fit type (tight/regular/loose) affects recommendations based on customer preference</span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="w-4 h-4 text-indigo-600 mt-0.5 flex-shrink-0" />
            <span>Gender-specific sizing is automatically matched to detected demographics</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

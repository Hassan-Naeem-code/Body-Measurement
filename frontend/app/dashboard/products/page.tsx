'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { brandAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import { toast } from 'sonner';
import type { Product, ProductRequest } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Dialog, DialogTrigger, DialogContent, DialogHeader } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Table, Thead, Tbody, Tr, Th, Td } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { EmptyState } from '@/components/ui/empty-state';

export default function ProductsPage() {
  const queryClient = useQueryClient();
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
  const productsQuery = useQuery<Product[]>({
    queryKey: ['products'],
    queryFn: async () => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return brandAPI.getProducts(apiKey);
    },
  });

  const addProductMutation = useMutation({
    mutationFn: async (payload: ProductRequest) => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');
      return brandAPI.addProduct(apiKey, payload);
    },
    onSuccess: () => {
      toast.success('Product added');
      setShowForm(false);
      queryClient.invalidateQueries({ queryKey: ['products'] });
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
    },
  });

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

  if (productsQuery.isLoading) {
    return (
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-40" />
            <Skeleton className="h-4 w-64 mt-2" />
          </div>
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="grid md:grid-cols-2 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-48" />
                <Skeleton className="h-4 w-32 mt-2" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-24 w-full" />
              </CardContent>
            </Card>
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
          <h1 className="text-3xl font-bold text-gray-900">Products</h1>
          <p className="text-gray-600 mt-2">Manage your product catalog and size charts</p>
        </div>
        <Dialog open={showForm} onOpenChange={setShowForm}>
          <DialogTrigger asChild>
            <Button>+ Add Product</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader title="Add New Product" />
            <form
              onSubmit={(e) => {
                e.preventDefault();
                addProductMutation.mutate(formData);
              }}
              className="space-y-4"
            >
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Product Name</label>
                  <Input
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="e.g., Classic T-Shirt"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Category</label>
                  <Input
                    type="text"
                    required
                    value={formData.category}
                    onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                    placeholder="e.g., T-Shirts, Jeans, Jackets"
                  />
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Size Chart (cm)</h3>
                <div className="overflow-x-auto">
                  <Table>
                    <Thead>
                      <Tr>
                        <Th>Size</Th>
                        <Th>Chest</Th>
                        <Th>Waist</Th>
                        <Th>Hip</Th>
                        <Th>Inseam</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {['XS', 'S', 'M', 'L', 'XL'].map((size) => (
                        <Tr key={size}>
                          <Td className="font-semibold">{size}</Td>
                          <Td>
                            <Input
                              type="number"
                              step="0.1"
                              value={formData.size_chart[size]?.chest || ''}
                              onChange={(e) => handleSizeChange(size, 'chest', e.target.value)}
                            />
                          </Td>
                          <Td>
                            <Input
                              type="number"
                              step="0.1"
                              value={formData.size_chart[size]?.waist || ''}
                              onChange={(e) => handleSizeChange(size, 'waist', e.target.value)}
                            />
                          </Td>
                          <Td>
                            <Input
                              type="number"
                              step="0.1"
                              value={formData.size_chart[size]?.hip || ''}
                              onChange={(e) => handleSizeChange(size, 'hip', e.target.value)}
                            />
                          </Td>
                          <Td>
                            <Input
                              type="number"
                              step="0.1"
                              value={formData.size_chart[size]?.inseam || ''}
                              onChange={(e) => handleSizeChange(size, 'inseam', e.target.value)}
                            />
                          </Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </div>
              </div>

              <Button type="submit" disabled={addProductMutation.isPending}>
                {addProductMutation.isPending ? 'Adding Product...' : 'Add Product'}
              </Button>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Error Display */}
      {productsQuery.error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {(productsQuery.error as Error)?.message}
        </div>
      )}

      {/* Products List */}

      <div className="space-y-4">
        {(productsQuery.data || []).length === 0 ? (
          <EmptyState
            title="No products yet"
            description="Add your first product with a size chart to start making recommendations."
            actionLabel="Add Your First Product"
            onAction={() => setShowForm(true)}
          />
        ) : (
          (productsQuery.data || []).map((product) => (
            <Card key={product.id}>
              <CardHeader>
                <CardTitle>{product.name}</CardTitle>
                <p className="text-sm text-gray-600">{product.category}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Added {new Date(product.created_at).toLocaleDateString()}
                </p>
              </CardHeader>
              <CardContent>
                <h4 className="text-sm font-semibold mb-2">Size Chart (cm)</h4>
                <div className="overflow-x-auto">
                  <Table>
                    <Thead>
                      <Tr>
                        <Th>Size</Th>
                        <Th>Chest</Th>
                        <Th>Waist</Th>
                        <Th>Hip</Th>
                        <Th>Inseam</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {product.size_chart &&
                        Object.entries(product.size_chart).map(([size, measurements]) => (
                          <Tr key={size}>
                            <Td className="font-semibold">{size}</Td>
                            <Td>{measurements.chest || '-'}</Td>
                            <Td>{measurements.waist || '-'}</Td>
                            <Td>{measurements.hip || '-'}</Td>
                            <Td>{measurements.inseam || '-'}</Td>
                          </Tr>
                        ))}
                      {!product.size_chart && (
                        <Tr>
                          <Td colSpan={5} className="text-center text-gray-500">
                            No size chart data
                          </Td>
                        </Tr>
                      )}
                    </Tbody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}

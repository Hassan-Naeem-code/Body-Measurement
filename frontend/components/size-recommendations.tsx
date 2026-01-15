'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { productsAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { BulkSizeRecommendation, PersonMeasurement } from '@/lib/types';
import { Shirt, Check, AlertCircle, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

interface SizeRecommendationsProps {
  measurements: PersonMeasurement;
}

const FIT_QUALITY_COLORS = {
  perfect: 'bg-green-100 text-green-700 border-green-200',
  good: 'bg-blue-100 text-blue-700 border-blue-200',
  acceptable: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  poor: 'bg-red-100 text-red-700 border-red-200',
};

const FIT_QUALITY_LABELS = {
  perfect: 'Perfect Fit',
  good: 'Good Fit',
  acceptable: 'Acceptable',
  poor: 'Poor Fit',
};

export function SizeRecommendations({ measurements }: SizeRecommendationsProps) {
  const [fitPreference, setFitPreference] = useState<'tight' | 'regular' | 'loose'>('regular');
  const [showAll, setShowAll] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | undefined>(undefined);

  const recommendationsQuery = useQuery({
    queryKey: ['size-recommendations', measurements.chest_circumference, measurements.waist_circumference, measurements.hip_circumference, measurements.estimated_height_cm, fitPreference, selectedCategory],
    queryFn: async () => {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) throw new Error('API key not found');

      return productsAPI.getBulkRecommendations(apiKey, {
        chest_circumference: measurements.chest_circumference || undefined,
        waist_circumference: measurements.waist_circumference || undefined,
        hip_circumference: measurements.hip_circumference || undefined,
        height: measurements.estimated_height_cm || undefined,
        fit_preference: fitPreference,
        category: selectedCategory,
      });
    },
    enabled: !!(measurements.chest_circumference || measurements.waist_circumference || measurements.hip_circumference),
  });

  const recommendations = recommendationsQuery.data?.recommendations || [];
  const displayedRecommendations = showAll ? recommendations : recommendations.slice(0, 4);

  if (!measurements.chest_circumference && !measurements.waist_circumference && !measurements.hip_circumference) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
            <Shirt className="w-5 h-5 text-indigo-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Size Recommendations</h3>
            <p className="text-sm text-gray-500">Based on your measurements</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="p-4 bg-gray-50 border-b border-gray-100">
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Fit Preference</label>
            <div className="flex gap-1">
              {(['tight', 'regular', 'loose'] as const).map((fit) => (
                <button
                  key={fit}
                  onClick={() => setFitPreference(fit)}
                  className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-colors ${
                    fitPreference === fit
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-200'
                  }`}
                >
                  {fit.charAt(0).toUpperCase() + fit.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Category</label>
            <select
              value={selectedCategory || ''}
              onChange={(e) => setSelectedCategory(e.target.value || undefined)}
              className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg bg-white"
            >
              <option value="">All Categories</option>
              <option value="tops">Tops</option>
              <option value="bottoms">Bottoms</option>
              <option value="dresses">Dresses</option>
              <option value="outerwear">Outerwear</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-5">
        {recommendationsQuery.isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg animate-pulse">
                <div className="w-10 h-10 bg-gray-200 rounded-lg" />
                <div className="flex-1">
                  <div className="h-4 w-32 bg-gray-200 rounded" />
                  <div className="h-3 w-20 bg-gray-200 rounded mt-2" />
                </div>
                <div className="h-8 w-16 bg-gray-200 rounded-lg" />
              </div>
            ))}
          </div>
        ) : recommendations.length === 0 ? (
          <div className="text-center py-8">
            <AlertCircle className="w-10 h-10 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No products with size charts found.</p>
            <p className="text-sm text-gray-400 mt-1">Add products with size charts to see recommendations.</p>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {displayedRecommendations.map((rec, index) => (
                <div
                  key={rec.product_id}
                  className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    index === 0 ? 'bg-indigo-100' : 'bg-gray-200'
                  }`}>
                    <Shirt className={`w-5 h-5 ${index === 0 ? 'text-indigo-600' : 'text-gray-500'}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium text-gray-900 truncate">{rec.product_name}</h4>
                      {index === 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-100 text-indigo-700 text-xs font-medium rounded-full">
                          <Sparkles className="w-3 h-3" />
                          Best Match
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-500 capitalize">{rec.category}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className={`px-3 py-1.5 rounded-lg text-sm font-medium border ${FIT_QUALITY_COLORS[rec.fit_quality]}`}>
                      {FIT_QUALITY_LABELS[rec.fit_quality]}
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900">{rec.recommended_size}</div>
                      <div className="text-xs text-gray-500">{Math.round(rec.confidence * 100)}% match</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {recommendations.length > 4 && (
              <button
                onClick={() => setShowAll(!showAll)}
                className="mt-4 w-full flex items-center justify-center gap-2 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                {showAll ? (
                  <>
                    <ChevronUp className="w-4 h-4" />
                    Show Less
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-4 h-4" />
                    Show {recommendations.length - 4} More Products
                  </>
                )}
              </button>
            )}
          </>
        )}
      </div>

      {/* Measurements Used */}
      <div className="px-5 pb-5">
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-xs font-medium text-gray-500 mb-2">Measurements Used</p>
          <div className="flex flex-wrap gap-3">
            {measurements.chest_circumference && (
              <span className="inline-flex items-center gap-1 text-sm">
                <Check className="w-3 h-3 text-green-600" />
                Chest: {measurements.chest_circumference.toFixed(1)}cm
              </span>
            )}
            {measurements.waist_circumference && (
              <span className="inline-flex items-center gap-1 text-sm">
                <Check className="w-3 h-3 text-green-600" />
                Waist: {measurements.waist_circumference.toFixed(1)}cm
              </span>
            )}
            {measurements.hip_circumference && (
              <span className="inline-flex items-center gap-1 text-sm">
                <Check className="w-3 h-3 text-green-600" />
                Hip: {measurements.hip_circumference.toFixed(1)}cm
              </span>
            )}
            {measurements.estimated_height_cm && (
              <span className="inline-flex items-center gap-1 text-sm">
                <Check className="w-3 h-3 text-green-600" />
                Height: {measurements.estimated_height_cm.toFixed(0)}cm
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

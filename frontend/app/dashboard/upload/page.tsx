'use client';

import { useState } from 'react';
import Image from 'next/image';
import { measurementsAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { MeasurementResult } from '@/lib/types';

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MeasurementResult | null>(null);
  const [error, setError] = useState('');

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError('');

    try {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) {
        setError('API key not found. Please log in again.');
        return;
      }

      const measurementResult = await measurementsAPI.processImage(apiKey, selectedFile);
      setResult(measurementResult);
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to process image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError('');
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Upload Image</h1>
        <p className="text-gray-600 mt-2">
          Upload a full-body photo to extract body measurements
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        {!previewUrl ? (
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-indigo-400 transition">
            <div className="text-6xl mb-4">📸</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Upload a photo
            </h3>
            <p className="text-gray-600 mb-4">
              Select a full-body photo for measurement analysis
            </p>
            <label className="inline-block">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <span className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 cursor-pointer inline-block transition">
                Choose File
              </span>
            </label>
            <p className="text-sm text-gray-500 mt-4">
              Supported formats: JPG, PNG, WEBP (Max 10MB)
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Selected Image</h3>
              <button
                onClick={handleReset}
                className="text-red-600 hover:text-red-700 font-semibold"
              >
                Remove
              </button>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <div className="relative w-full h-96">
                  <Image
                    src={previewUrl}
                    alt="Preview"
                    fill
                    className="rounded-lg border border-gray-200 object-contain"
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  {selectedFile?.name} ({(selectedFile?.size || 0 / 1024 / 1024).toFixed(2)} MB)
                </p>
              </div>
              <div className="flex flex-col justify-center">
                <h4 className="font-semibold text-gray-900 mb-4">
                  Ready to process
                </h4>
                <button
                  onClick={handleUpload}
                  disabled={loading}
                  className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition disabled:bg-indigo-400 disabled:cursor-not-allowed"
                >
                  {loading ? 'Processing...' : 'Process Measurements'}
                </button>
                {loading && (
                  <p className="text-sm text-gray-600 mt-4">
                    Analyzing image with AI... This may take a few seconds.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Recommended Size */}
          <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-6 rounded-lg text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100">Recommended Size</p>
                <p className="text-4xl font-bold mt-2">{result.recommended_size}</p>
                <p className="text-green-100 mt-2">
                  Processed in {result.processing_time_ms}ms
                </p>
              </div>
              <div className="text-6xl">✓</div>
            </div>
          </div>

          {/* Measurements Grid */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Body Measurements</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Shoulder Width</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {result.shoulder_width.toFixed(1)} cm
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Confidence: {(result.confidence_scores.shoulder_width * 100).toFixed(0)}%
                </p>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Chest Width</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {result.chest_width.toFixed(1)} cm
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Confidence: {(result.confidence_scores.chest_width * 100).toFixed(0)}%
                </p>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Waist Width</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {result.waist_width.toFixed(1)} cm
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Confidence: {(result.confidence_scores.waist_width * 100).toFixed(0)}%
                </p>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Hip Width</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {result.hip_width.toFixed(1)} cm
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Confidence: {(result.confidence_scores.hip_width * 100).toFixed(0)}%
                </p>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Inseam</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {result.inseam.toFixed(1)} cm
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Confidence: {(result.confidence_scores.inseam * 100).toFixed(0)}%
                </p>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-600">Arm Length</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {result.arm_length.toFixed(1)} cm
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Confidence: {(result.confidence_scores.arm_length * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </div>

          {/* Size Probabilities */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Size Distribution</h3>
            <div className="space-y-3">
              {Object.entries(result.size_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([size, probability]) => (
                  <div key={size}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-gray-900">Size {size}</span>
                      <span className="text-gray-600">{(probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 rounded-full h-2"
                        style={{ width: `${probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-4">
            <button
              onClick={handleReset}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
            >
              Upload Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

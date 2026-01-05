'use client';

import { useState } from 'react';
import Image from 'next/image';
import { measurementsAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { MultiPersonMeasurementResult } from '@/lib/types';

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MultiPersonMeasurementResult | null>(null);
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

      const measurementResult = await measurementsAPI.processMultiPerson(apiKey, selectedFile);
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
          {/* Summary Banner */}
          <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-6 rounded-lg text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100">Multi-Person Detection Results</p>
                <p className="text-4xl font-bold mt-2">
                  {result.valid_people_count} {result.valid_people_count === 1 ? 'Person' : 'People'} Detected
                </p>
                <p className="text-green-100 mt-2">
                  Total detected: {result.total_people_detected} | Valid: {result.valid_people_count} | Invalid: {result.invalid_people_count}
                </p>
                <p className="text-green-100 mt-1">
                  Processed in {result.processing_time_ms.toFixed(0)}ms
                </p>
              </div>
              <div className="text-6xl">✓</div>
            </div>
          </div>

          {/* Display Each Person */}
          {result.measurements.map((person, index) => (
            <div key={person.person_id} className="bg-white p-6 rounded-lg shadow-sm border-2 border-indigo-200">
              {/* Person Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-900">
                    Person {person.person_id + 1}
                    {person.demographic_label && (
                      <span className="ml-3 text-lg font-semibold text-indigo-600">
                        ({person.demographic_label})
                      </span>
                    )}
                  </h3>
                  <div className="flex gap-4 mt-2 text-sm">
                    <span className="text-gray-600">
                      Detection: {(person.detection_confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-gray-600">
                      Validation: {(person.validation_confidence * 100).toFixed(0)}%
                    </span>
                    {person.estimated_height_cm && (
                      <span className="text-gray-600">
                        Estimated Height: {person.estimated_height_cm.toFixed(1)} cm
                      </span>
                    )}
                  </div>
                  {person.gender && person.age_group && (
                    <div className="flex gap-3 mt-2">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                        {person.gender === 'male' ? '♂️ Male' : '♀️ Female'} ({(person.gender_confidence! * 100).toFixed(0)}%)
                      </span>
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {person.age_group.charAt(0).toUpperCase() + person.age_group.slice(1)} ({(person.age_confidence! * 100).toFixed(0)}%)
                      </span>
                    </div>
                  )}
                </div>
                {person.is_valid ? (
                  <span className="bg-green-100 text-green-800 px-4 py-2 rounded-full font-semibold">
                    ✓ Valid
                  </span>
                ) : (
                  <span className="bg-red-100 text-red-800 px-4 py-2 rounded-full font-semibold">
                    ✗ Invalid
                  </span>
                )}
              </div>

              {person.is_valid && person.recommended_size ? (
                <>
                  {/* Recommended Size */}
                  <div className="bg-indigo-50 p-4 rounded-lg mb-6">
                    <p className="text-sm text-indigo-600 font-semibold">RECOMMENDED SIZE</p>
                    <p className="text-3xl font-bold text-indigo-900 mt-1">{person.recommended_size}</p>
                  </div>

                  {/* Circumference Measurements (95%+ Accuracy) */}
                  <div className="mb-6">
                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-bold">95%+ ACCURACY</span>
                      Circumference Measurements
                    </h4>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {person.chest_circumference && (
                        <div className="p-4 bg-green-50 rounded-lg border-2 border-green-200">
                          <p className="text-sm text-gray-600">Chest Circumference</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.chest_circumference.toFixed(1)} cm
                          </p>
                          <p className="text-xs text-green-700 mt-1">◉ Ellipse Formula</p>
                        </div>
                      )}

                      {person.waist_circumference && (
                        <div className="p-4 bg-green-50 rounded-lg border-2 border-green-200">
                          <p className="text-sm text-gray-600">Waist Circumference</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.waist_circumference.toFixed(1)} cm
                          </p>
                          <p className="text-xs text-green-700 mt-1">◉ Ellipse Formula</p>
                        </div>
                      )}

                      {person.hip_circumference && (
                        <div className="p-4 bg-green-50 rounded-lg border-2 border-green-200">
                          <p className="text-sm text-gray-600">Hip Circumference</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.hip_circumference.toFixed(1)} cm
                          </p>
                          <p className="text-xs text-green-700 mt-1">◉ Ellipse Formula</p>
                        </div>
                      )}

                      {person.arm_circumference && (
                        <div className="p-4 bg-green-50 rounded-lg border-2 border-green-200">
                          <p className="text-sm text-gray-600">Arm Circumference</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.arm_circumference.toFixed(1)} cm
                          </p>
                          <p className="text-xs text-green-700 mt-1">◉ Ellipse Formula</p>
                        </div>
                      )}

                      {person.thigh_circumference && (
                        <div className="p-4 bg-green-50 rounded-lg border-2 border-green-200">
                          <p className="text-sm text-gray-600">Thigh Circumference</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.thigh_circumference.toFixed(1)} cm
                          </p>
                          <p className="text-xs text-green-700 mt-1">◉ Ellipse Formula</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Width Measurements (Reference) */}
                  <div className="mb-6">
                    <h4 className="font-semibold text-gray-900 mb-3">Width Measurements (Reference)</h4>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {person.shoulder_width && (
                        <div className="p-4 bg-blue-50 rounded-lg">
                          <p className="text-sm text-gray-600">Shoulder Width</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.shoulder_width.toFixed(1)} cm
                          </p>
                        </div>
                      )}

                      {person.inseam && (
                        <div className="p-4 bg-blue-50 rounded-lg">
                          <p className="text-sm text-gray-600">Inseam</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.inseam.toFixed(1)} cm
                          </p>
                        </div>
                      )}

                      {person.arm_length && (
                        <div className="p-4 bg-blue-50 rounded-lg">
                          <p className="text-sm text-gray-600">Arm Length</p>
                          <p className="text-2xl font-bold text-gray-900 mt-1">
                            {person.arm_length.toFixed(1)} cm
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Size Probabilities */}
                  {person.size_probabilities && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Size Distribution</h4>
                      <div className="space-y-2">
                        {Object.entries(person.size_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([size, probability]) => (
                            <div key={size}>
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-sm font-semibold text-gray-900">Size {size}</span>
                                <span className="text-sm text-gray-600">{(probability * 100).toFixed(1)}%</span>
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
                  )}
                </>
              ) : (
                <div className="bg-red-50 p-4 rounded-lg">
                  <p className="text-red-800 font-semibold mb-2">
                    This person was filtered out because:
                  </p>
                  <ul className="list-disc list-inside text-red-700 space-y-1">
                    {person.missing_parts.map((part, idx) => (
                      <li key={idx}>{part.replace(/_/g, ' ')}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}

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

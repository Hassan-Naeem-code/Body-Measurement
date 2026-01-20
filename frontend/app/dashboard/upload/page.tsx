'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Image from 'next/image';
import { measurementsAPI } from '@/lib/api';
import { authHelpers } from '@/lib/auth';
import type { MultiPersonMeasurementResult } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import axios from 'axios';
import { useNavigationGuardContext } from '@/contexts/NavigationGuardContext';
import {
  Upload,
  X,
  Camera,
  FileImage,
  AlertCircle,
  CheckCircle2,
  User,
  Loader2,
  Ruler,
  Sparkles,
  Clock,
  Users,
  BadgeCheck,
  AlertTriangle,
  Eye,
  EyeOff,
} from 'lucide-react';
import { MeasurementVisualization } from '@/components/measurement-visualization';
import { PDFExportButton } from '@/components/pdf-export-button';
import { CameraCapture } from '@/components/camera-capture';
import { SizeRecommendations } from '@/components/size-recommendations';

// Utility function to format file size
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Utility function to format processing time
const formatTime = (ms: number): string => {
  if (ms >= 1000) {
    return (ms / 1000).toFixed(2) + 's';
  }
  return Math.round(ms) + 'ms';
};

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MultiPersonMeasurementResult | null>(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);

  // Visualization options
  const [showVisualization, setShowVisualization] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showMeasurements, setShowMeasurements] = useState(true);
  const [showBoundingBox, setShowBoundingBox] = useState(true);
  const [selectedPersonId, setSelectedPersonId] = useState<number | undefined>(undefined);

  // Camera capture
  const [showCamera, setShowCamera] = useState(false);

  // Navigation guard context
  const { setNavigationBlocked } = useNavigationGuardContext();

  // AbortController ref for cancelling requests
  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup function to cancel pending requests
  const cancelPendingRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    measurementsAPI.cancelPending();
  }, []);

  // Cancel requests on component unmount
  useEffect(() => {
    return () => {
      cancelPendingRequest();
    };
  }, [cancelPendingRequest]);

  // Cleanup Object URL to prevent memory leaks
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Block navigation when processing is in progress
  useEffect(() => {
    setNavigationBlocked(
      loading,
      'Your image is still being processed. If you leave now, you will lose your results. Are you sure you want to leave?'
    );

    // Cleanup: unblock navigation when component unmounts
    return () => {
      setNavigationBlocked(false);
    };
  }, [loading, setNavigationBlocked]);

  // Handle drag events
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  // Handle drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  // Handle file selection
  const handleFile = (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size must be less than 10MB');
      return;
    }

    cancelPendingRequest();
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError('');
    setLoading(false);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    cancelPendingRequest();
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setError('');

    try {
      const apiKey = authHelpers.getApiKey();
      if (!apiKey) {
        setError('API key not found. Please log in again.');
        return;
      }

      const measurementResult = await measurementsAPI.processMultiPerson(
        apiKey,
        selectedFile,
        abortControllerRef.current.signal
      );
      setResult(measurementResult);
      toast.success('Image processed successfully');
    } catch (err: unknown) {
      if (axios.isCancel(err) || (err instanceof Error && err.name === 'AbortError')) {
        // Request was cancelled by user - no action needed
        return;
      }

      const error = err as { response?: { data?: { detail?: string } }; message?: string };
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to process image. Please try again.';
      setError(errorMessage);
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleReset = () => {
    cancelPendingRequest();
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError('');
    setLoading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl lg:text-3xl font-bold tracking-tight text-foreground">
            Upload Image
          </h1>
          <p className="text-muted-foreground">
            Upload a full-body photo to extract body measurements
          </p>
        </div>
      </div>

      {/* Upload Section */}
      <div className="rounded-2xl border border-border bg-card overflow-hidden shadow-card">
        {!previewUrl ? (
          <div
            className={`relative p-8 lg:p-12 transition-all duration-200 ${
              dragActive
                ? 'bg-primary-muted border-2 border-dashed border-primary'
                : 'border-2 border-dashed border-border hover:border-primary/50 hover:bg-muted/30'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center text-center space-y-6">
              <div
                className={`w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-200 ${
                  dragActive ? 'bg-primary scale-110' : 'bg-primary-muted'
                }`}
              >
                <Camera
                  className={`w-10 h-10 transition-colors ${
                    dragActive ? 'text-primary-foreground' : 'text-primary'
                  }`}
                />
              </div>

              <div className="space-y-2">
                <h3 className="text-xl font-semibold text-foreground">
                  {dragActive ? 'Drop your image here' : 'Upload a photo'}
                </h3>
                <p className="text-muted-foreground max-w-md">
                  Drag and drop a full-body image here, or click to browse
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3">
                <label className="cursor-pointer">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <span className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary-hover active:bg-primary-active transition-all duration-200 shadow-sm hover:shadow-md">
                    <Upload className="w-5 h-5" />
                    Select Image
                  </span>
                </label>

                <button
                  onClick={() => setShowCamera(true)}
                  className="inline-flex items-center gap-2 px-6 py-3 border-2 border-primary text-primary rounded-xl font-medium hover:bg-primary-muted transition-all duration-200"
                >
                  <Camera className="w-5 h-5" />
                  Use Camera
                </button>
              </div>

              <div className="flex items-center gap-6 text-sm text-muted-foreground">
                <span className="flex items-center gap-2">
                  <FileImage className="w-4 h-4" />
                  JPG, PNG, WEBP
                </span>
                <span className="flex items-center gap-2">
                  <Ruler className="w-4 h-4" />
                  Max 10MB
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-6 lg:p-8 space-y-6">
            {/* Image Preview Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-success-muted flex items-center justify-center">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Image Selected</h3>
                  <p className="text-sm text-muted-foreground">
                    {selectedFile?.name} ({formatFileSize(selectedFile?.size || 0)})
                  </p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-destructive hover:bg-destructive-muted transition-colors"
              >
                <X className="w-4 h-4" />
                Remove
              </button>
            </div>

            {/* Preview Grid */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Image Preview */}
              <div className="relative aspect-[3/4] rounded-xl overflow-hidden border border-border bg-muted">
                <Image
                  src={previewUrl}
                  alt="Preview"
                  fill
                  className="object-contain"
                />
              </div>

              {/* Action Panel */}
              <div className="flex flex-col justify-center space-y-6">
                <div className="p-6 rounded-xl bg-muted/50 border border-border space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">Ready to Process</h4>
                      <p className="text-sm text-muted-foreground">AI will analyze body measurements</p>
                    </div>
                  </div>

                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-success" />
                      Multi-person detection
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-success" />
                      Gender & age estimation
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-success" />
                      Accurate circumferences
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-success" />
                      Size recommendations
                    </li>
                  </ul>
                </div>

                <Button
                  onClick={handleUpload}
                  disabled={loading}
                  className="w-full h-14 text-base font-semibold rounded-xl"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 mr-2" />
                      Process Measurements
                    </>
                  )}
                </Button>

                {loading && (
                  <div className="flex items-center gap-3 p-4 rounded-xl bg-info-muted border border-info/20">
                    <Clock className="w-5 h-5 text-info animate-pulse" />
                    <p className="text-sm text-foreground">
                      Analyzing image with AI... This may take a few seconds.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="flex items-start gap-4 p-6 rounded-2xl bg-destructive-muted border border-destructive/20 animate-slide-up">
          <div className="w-10 h-10 rounded-xl bg-destructive/10 flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-5 h-5 text-destructive" />
          </div>
          <div>
            <h4 className="font-semibold text-foreground">Processing Failed</h4>
            <p className="text-muted-foreground mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6 animate-slide-up">
          {/* Visualization Section */}
          {previewUrl && result.measurements.length > 0 && (
            <div className="rounded-2xl border border-border bg-card overflow-hidden shadow-card">
              {/* Visualization Header */}
              <div className="p-4 border-b border-border bg-muted/30 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                    <Eye className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">Body Measurement Visualization</h3>
                    <p className="text-sm text-muted-foreground">See where measurements were taken</p>
                  </div>
                </div>

                {/* Toggle Controls */}
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => setShowVisualization(!showVisualization)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      showVisualization
                        ? 'bg-primary text-white'
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    {showVisualization ? 'Hide' : 'Show'} Overlay
                  </button>
                  {showVisualization && (
                    <>
                      <button
                        onClick={() => setShowSkeleton(!showSkeleton)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          showSkeleton
                            ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
                            : 'bg-muted text-muted-foreground hover:bg-muted/80'
                        }`}
                      >
                        Skeleton
                      </button>
                      <button
                        onClick={() => setShowMeasurements(!showMeasurements)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          showMeasurements
                            ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
                            : 'bg-muted text-muted-foreground hover:bg-muted/80'
                        }`}
                      >
                        Measurements
                      </button>
                      <button
                        onClick={() => setShowBoundingBox(!showBoundingBox)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          showBoundingBox
                            ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                            : 'bg-muted text-muted-foreground hover:bg-muted/80'
                        }`}
                      >
                        Bounding Box
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Visualization Canvas */}
              <div className="p-4">
                {showVisualization ? (
                  <MeasurementVisualization
                    imageUrl={previewUrl}
                    measurements={result.measurements}
                    selectedPersonId={selectedPersonId}
                    showSkeleton={showSkeleton}
                    showMeasurements={showMeasurements}
                    showBoundingBox={showBoundingBox}
                  />
                ) : (
                  <div className="relative w-full aspect-auto">
                    <img
                      src={previewUrl}
                      alt="Uploaded image"
                      className="w-full h-auto rounded-lg max-h-[80vh] object-contain"
                    />
                  </div>
                )}

                {/* Person Selector (if multiple people) */}
                {result.measurements.length > 1 && (
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <span className="text-sm text-muted-foreground mr-2">Show person:</span>
                    <button
                      onClick={() => setSelectedPersonId(undefined)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                        selectedPersonId === undefined
                          ? 'bg-primary text-white'
                          : 'bg-muted text-muted-foreground hover:bg-muted/80'
                      }`}
                    >
                      All
                    </button>
                    {result.measurements.map((person) => (
                      <button
                        key={person.person_id}
                        onClick={() => setSelectedPersonId(person.person_id)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          selectedPersonId === person.person_id
                            ? 'bg-primary text-white'
                            : 'bg-muted text-muted-foreground hover:bg-muted/80'
                        }`}
                      >
                        Person {person.person_id + 1}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Summary Banner */}
          <div className="relative overflow-hidden rounded-2xl gradient-success p-6 lg:p-8 text-white">
            <div className="absolute inset-0 opacity-10">
              <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2" />
            </div>

            <div className="relative flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="space-y-2">
                <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-sm font-medium backdrop-blur-sm">
                  <BadgeCheck className="w-4 h-4" />
                  <span>Processing Complete</span>
                </div>
                <h2 className="text-3xl lg:text-4xl font-bold">
                  {result.valid_people_count} {result.valid_people_count === 1 ? 'Person' : 'People'} Detected
                </h2>
                <div className="flex flex-wrap items-center gap-4 text-sm text-white/80">
                  <span className="flex items-center gap-1.5">
                    <Users className="w-4 h-4" />
                    Total: {result.total_people_detected}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    Valid: {result.valid_people_count}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4" />
                    Invalid: {result.invalid_people_count}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Clock className="w-4 h-4" />
                    {formatTime(result.processing_time_ms)}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <PDFExportButton
                  result={result}
                  variant="secondary"
                  className="bg-white/20 hover:bg-white/30 border-white/30 text-white"
                />
                <div className="hidden sm:flex items-center justify-center w-20 h-20 rounded-2xl bg-white/10 backdrop-blur-sm">
                  <CheckCircle2 className="w-10 h-10" />
                </div>
              </div>
            </div>
          </div>

          {/* Display Each Person */}
          {result.measurements.map((person) => (
            <div
              key={person.person_id}
              className="rounded-2xl border-2 border-border bg-card overflow-hidden shadow-card"
            >
              {/* Person Header */}
              <div className="p-6 border-b border-border bg-muted/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div
                      className={`w-14 h-14 rounded-2xl flex items-center justify-center ${
                        person.is_valid ? 'gradient-primary' : 'bg-destructive-muted'
                      }`}
                    >
                      <User
                        className={`w-7 h-7 ${
                          person.is_valid ? 'text-white' : 'text-destructive'
                        }`}
                      />
                    </div>
                    <div>
                      <div className="flex items-center gap-3">
                        <h3 className="text-xl font-bold text-foreground">
                          Person {person.person_id + 1}
                        </h3>
                        {person.demographic_label && (
                          <span className="badge badge-primary">{person.demographic_label}</span>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-3 mt-2 text-sm text-muted-foreground">
                        <span>Detection: {(person.detection_confidence * 100).toFixed(0)}%</span>
                        <span>Validation: {(person.validation_confidence * 100).toFixed(0)}%</span>
                        {person.estimated_height_cm && (
                          <span>Height: {person.estimated_height_cm.toFixed(1)} cm</span>
                        )}
                      </div>
                      {person.gender && person.age_group && (
                        <div className="flex gap-2 mt-2">
                          <span className="badge bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                            {person.gender === 'male' ? 'Male' : 'Female'} ({(person.gender_confidence! * 100).toFixed(0)}%)
                          </span>
                          <span className="badge bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                            {person.age_group.charAt(0).toUpperCase() + person.age_group.slice(1)} ({(person.age_confidence! * 100).toFixed(0)}%)
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                  <span
                    className={`badge px-4 py-2 text-sm font-semibold ${
                      person.is_valid
                        ? 'bg-success-muted text-success-foreground'
                        : 'bg-destructive-muted text-destructive'
                    }`}
                  >
                    {person.is_valid ? 'Valid' : 'Invalid'}
                  </span>
                </div>
              </div>

              {person.is_valid && person.recommended_size ? (
                <div className="p-6 space-y-6">
                  {/* Recommended Size */}
                  <div className="p-6 rounded-xl bg-primary-muted border border-primary/20">
                    <p className="text-sm font-medium text-primary uppercase tracking-wide">
                      Recommended Size
                    </p>
                    <p className="text-5xl font-bold text-primary mt-2">
                      {person.recommended_size}
                    </p>
                  </div>

                  {/* Circumference Measurements */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <span className="badge badge-success">95%+ ACCURACY</span>
                      <h4 className="font-semibold text-foreground">Circumference Measurements</h4>
                    </div>
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {[
                        { label: 'Chest', value: person.chest_circumference },
                        { label: 'Waist', value: person.waist_circumference },
                        { label: 'Hip', value: person.hip_circumference },
                        { label: 'Arm', value: person.arm_circumference },
                        { label: 'Thigh', value: person.thigh_circumference },
                      ].map(
                        (measurement) =>
                          measurement.value && (
                            <div
                              key={measurement.label}
                              className="p-4 rounded-xl bg-success-muted/50 border border-success/20"
                            >
                              <p className="text-sm text-muted-foreground">{measurement.label} Circumference</p>
                              <p className="text-2xl font-bold text-foreground mt-1">
                                {measurement.value.toFixed(1)} <span className="text-sm font-normal text-muted-foreground">cm</span>
                              </p>
                            </div>
                          )
                      )}
                    </div>
                  </div>

                  {/* Width Measurements */}
                  <div className="space-y-4">
                    <h4 className="font-semibold text-foreground">Width Measurements (Reference)</h4>
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {[
                        { label: 'Shoulder Width', value: person.shoulder_width },
                        { label: 'Inseam', value: person.inseam },
                        { label: 'Arm Length', value: person.arm_length },
                      ].map(
                        (measurement) =>
                          measurement.value && (
                            <div
                              key={measurement.label}
                              className="p-4 rounded-xl bg-muted/50 border border-border"
                            >
                              <p className="text-sm text-muted-foreground">{measurement.label}</p>
                              <p className="text-2xl font-bold text-foreground mt-1">
                                {measurement.value.toFixed(1)} <span className="text-sm font-normal text-muted-foreground">cm</span>
                              </p>
                            </div>
                          )
                      )}
                    </div>
                  </div>

                  {/* Size Probabilities */}
                  {person.size_probabilities && (
                    <div className="space-y-4">
                      <h4 className="font-semibold text-foreground">Size Distribution</h4>
                      <div className="space-y-3">
                        {Object.entries(person.size_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([size, probability]) => (
                            <div key={size}>
                              <div className="flex items-center justify-between mb-1.5">
                                <span className="text-sm font-medium text-foreground">Size {size}</span>
                                <span className="text-sm text-muted-foreground">{(probability * 100).toFixed(1)}%</span>
                              </div>
                              <div className="progress-bar">
                                <div
                                  className="progress-bar-fill"
                                  style={{ width: `${probability * 100}%` }}
                                />
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Product Size Recommendations */}
                  <SizeRecommendations measurements={person} />
                </div>
              ) : (
                <div className="p-6">
                  <div className="p-6 rounded-xl bg-destructive-muted/50 border border-destructive/20">
                    <div className="flex items-start gap-4">
                      <AlertTriangle className="w-6 h-6 text-destructive flex-shrink-0" />
                      <div>
                        <p className="font-semibold text-foreground">
                          This person was filtered out because:
                        </p>
                        <ul className="mt-2 space-y-1 text-muted-foreground">
                          {person.missing_parts.map((part, idx) => (
                            <li key={idx} className="flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-destructive" />
                              {part.replace(/_/g, ' ')}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4">
            <Button onClick={handleReset} variant="outline" className="flex-1 sm:flex-none">
              <Upload className="w-4 h-4 mr-2" />
              Upload Another Image
            </Button>
            <PDFExportButton result={result} />
          </div>
        </div>
      )}

      {/* Camera Capture Modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleFile}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}

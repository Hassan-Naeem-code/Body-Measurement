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
  Box,
  RotateCcw,
} from 'lucide-react';
import { MeasurementVisualization } from '@/components/measurement-visualization';
import { PDFExportButton } from '@/components/pdf-export-button';
import { CameraCapture } from '@/components/camera-capture';
import { SizeRecommendations } from '@/components/size-recommendations';
import { SizeResultCard } from '@/components/size-result-card';
import Body3DViewer from '@/components/body-3d-viewer';

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

// Utility function to crop an image based on a bounding box
const cropImageToBoundingBox = (
  imageUrl: string,
  boundingBox: { x1: number; y1: number; x2: number; y2: number },
  padding: number = 0.1 // 10% padding around the person
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = document.createElement('img');
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }

      // Convert normalized coordinates (0-1) to pixel coordinates
      const imgWidth = img.naturalWidth;
      const imgHeight = img.naturalHeight;

      const x1 = boundingBox.x1 * imgWidth;
      const y1 = boundingBox.y1 * imgHeight;
      const x2 = boundingBox.x2 * imgWidth;
      const y2 = boundingBox.y2 * imgHeight;

      // Calculate crop dimensions with padding
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;
      const paddingX = boxWidth * padding;
      const paddingY = boxHeight * padding;

      // Apply padding but keep within image bounds
      const cropX = Math.max(0, x1 - paddingX);
      const cropY = Math.max(0, y1 - paddingY);
      const cropWidth = Math.min(imgWidth - cropX, boxWidth + 2 * paddingX);
      const cropHeight = Math.min(imgHeight - cropY, boxHeight + 2 * paddingY);

      // Set canvas size to the crop dimensions
      canvas.width = cropWidth;
      canvas.height = cropHeight;

      // Draw the cropped portion
      ctx.drawImage(
        img,
        cropX, cropY, cropWidth, cropHeight, // Source rectangle
        0, 0, cropWidth, cropHeight // Destination rectangle
      );

      // Convert to blob URL
      canvas.toBlob((blob) => {
        if (blob) {
          const croppedUrl = URL.createObjectURL(blob);
          resolve(croppedUrl);
        } else {
          reject(new Error('Failed to create cropped image'));
        }
      }, 'image/png');
    };
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = imageUrl;
  });
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

  // 3D Viewer
  const [show3DViewer, setShow3DViewer] = useState(false);
  const [viewer3DPerson, setViewer3DPerson] = useState<{
    height: number;
    gender: 'male' | 'female' | 'neutral';
    chest: number;
    waist: number;
    hip: number;
    imageUrl?: string; // URL of the uploaded image for texture mapping
  } | null>(null);
  const [viewer3DMode, setViewer3DMode] = useState<'mannequin' | 'texture' | 'depth2.5d' | 'pifuhd'>('mannequin');

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

  // Cleanup cropped image URL when 3D viewer closes or changes
  const croppedImageUrlRef = useRef<string | null>(null);
  useEffect(() => {
    // If imageUrl is a blob URL (starts with blob:) and different from previewUrl, it's a cropped image
    if (viewer3DPerson?.imageUrl &&
        viewer3DPerson.imageUrl.startsWith('blob:') &&
        viewer3DPerson.imageUrl !== previewUrl) {
      // Revoke previous cropped URL if it exists
      if (croppedImageUrlRef.current && croppedImageUrlRef.current !== viewer3DPerson.imageUrl) {
        URL.revokeObjectURL(croppedImageUrlRef.current);
      }
      croppedImageUrlRef.current = viewer3DPerson.imageUrl;
    }

    return () => {
      // Cleanup on unmount
      if (croppedImageUrlRef.current) {
        URL.revokeObjectURL(croppedImageUrlRef.current);
        croppedImageUrlRef.current = null;
      }
    };
  }, [viewer3DPerson?.imageUrl, previewUrl]);

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
    setShow3DViewer(false);
    setViewer3DPerson(null);
    // Cleanup cropped image URL
    if (croppedImageUrlRef.current) {
      URL.revokeObjectURL(croppedImageUrlRef.current);
      croppedImageUrlRef.current = null;
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Open 3D viewer for a person
  const handleView3D = async (person: {
    estimated_height_cm?: number | null;
    gender?: string | null;
    chest_circumference?: number | null;
    waist_circumference?: number | null;
    hip_circumference?: number | null;
    bounding_box?: { x1: number; y1: number; x2: number; y2: number } | null;
  }) => {
    let imageUrl: string | undefined = undefined;

    // If we have a bounding box, crop the image to show only this person
    if (previewUrl && person.bounding_box) {
      try {
        imageUrl = await cropImageToBoundingBox(previewUrl, person.bounding_box, 0.15);
      } catch (err) {
        console.error('Failed to crop image:', err);
        // Fall back to full image if cropping fails
        imageUrl = previewUrl;
      }
    } else if (previewUrl) {
      // No bounding box available, use full image
      imageUrl = previewUrl;
    }

    setViewer3DPerson({
      height: person.estimated_height_cm ?? 170,
      gender: (person.gender === 'male' || person.gender === 'female') ? person.gender : 'neutral',
      chest: person.chest_circumference ?? 95,
      waist: person.waist_circumference ?? 80,
      hip: person.hip_circumference ?? 95,
      imageUrl,
    });
    setShow3DViewer(true);
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

          {/* Display Each Person - Clean Size Result Cards */}
          {result.measurements.map((person) => (
            <div key={person.person_id} className="space-y-3">
              <SizeResultCard person={person} personIndex={person.person_id} />

              {/* Additional actions for valid persons */}
              {person.is_valid && (
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => handleView3D(person)}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-indigo-100 text-indigo-700 hover:bg-indigo-200 dark:bg-indigo-900/30 dark:text-indigo-300 dark:hover:bg-indigo-900/50 transition-colors"
                  >
                    <Box className="w-4 h-4" />
                    View 3D Body Model
                  </button>

                  {/* Product Size Recommendations - Expandable */}
                  <details className="w-full mt-2">
                    <summary className="cursor-pointer px-4 py-2 rounded-lg text-sm font-medium bg-muted hover:bg-muted/80 transition-colors inline-flex items-center gap-2">
                      <Ruler className="w-4 h-4" />
                      View Product-Specific Sizes
                    </summary>
                    <div className="mt-3">
                      <SizeRecommendations measurements={person} />
                    </div>
                  </details>
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

      {/* 3D Body Viewer Modal */}
      {show3DViewer && viewer3DPerson && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
          <div className="relative w-full max-w-5xl bg-card rounded-2xl overflow-hidden shadow-2xl border border-border">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-border bg-muted/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                  <Box className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">3D Body View</h3>
                  <p className="text-sm text-muted-foreground">
                    Drag to rotate • Scroll to zoom • Based on your measurements
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShow3DViewer(false)}
                className="p-2 rounded-lg hover:bg-muted transition-colors"
              >
                <X className="w-5 h-5 text-muted-foreground" />
              </button>
            </div>

            {/* 3D Viewer */}
            <div className="relative">
              <Body3DViewer
                heightCm={viewer3DPerson.height}
                gender={viewer3DPerson.gender}
                showControls={true}
                showMeasurements={true}
                autoRotate={true}
                className="h-[70vh] min-h-[500px]"
                imageUrl={viewer3DPerson.imageUrl}
                viewMode={viewer3DMode}
                onViewModeChange={setViewer3DMode}
              />
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-border bg-muted/30 flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                <span className="font-medium text-foreground">Your measurements:</span>{' '}
                Height {viewer3DPerson.height}cm • Chest {viewer3DPerson.chest}cm • Waist {viewer3DPerson.waist}cm • Hip {viewer3DPerson.hip}cm
              </div>
              <Button
                onClick={() => setShow3DViewer(false)}
                variant="outline"
              >
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

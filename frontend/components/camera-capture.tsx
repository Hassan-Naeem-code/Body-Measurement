'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Camera,
  X,
  SwitchCamera,
  ZoomIn,
  ZoomOut,
  Check,
  AlertCircle,
  Loader2,
  RotateCcw,
} from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  onClose: () => void;
}

interface CameraConstraints {
  facingMode: 'user' | 'environment';
  zoom: number;
}

export function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [constraints, setConstraints] = useState<CameraConstraints>({
    facingMode: 'environment',
    zoom: 1,
  });
  const [capabilities, setCapabilities] = useState<{
    hasFrontCamera: boolean;
    hasBackCamera: boolean;
    supportsZoom: boolean;
    maxZoom: number;
  }>({
    hasFrontCamera: false,
    hasBackCamera: false,
    supportsZoom: false,
    maxZoom: 1,
  });

  // Initialize camera
  const initializeCamera = useCallback(async () => {
    setIsInitializing(true);
    setError(null);

    try {
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      // Check for camera support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported on this browser');
      }

      // Get stream with constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: constraints.facingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });

      streamRef.current = stream;
      setHasPermission(true);

      // Check capabilities
      const videoTrack = stream.getVideoTracks()[0];
      const trackCapabilities = videoTrack.getCapabilities?.() || {};

      setCapabilities({
        hasFrontCamera: true, // We'll assume both if we got this far
        hasBackCamera: true,
        supportsZoom: 'zoom' in trackCapabilities,
        maxZoom: (trackCapabilities as { zoom?: { max: number } }).zoom?.max || 1,
      });

      // Apply zoom if supported
      if ('zoom' in trackCapabilities && constraints.zoom > 1) {
        const settings = { zoom: constraints.zoom };
        await videoTrack.applyConstraints({ advanced: [settings as MediaTrackConstraintSet] });
      }

      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error('Camera error:', err);
      const error = err as Error;

      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        setHasPermission(false);
        setError('Camera access denied. Please allow camera access in your browser settings.');
      } else if (error.name === 'NotFoundError') {
        setError('No camera found on this device.');
      } else if (error.name === 'NotReadableError') {
        setError('Camera is in use by another application.');
      } else {
        setError(error.message || 'Failed to access camera');
      }
    } finally {
      setIsInitializing(false);
    }
  }, [constraints.facingMode, constraints.zoom]);

  // Initialize on mount
  useEffect(() => {
    initializeCamera();

    return () => {
      // Cleanup on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [initializeCamera]);

  // Switch camera
  const switchCamera = useCallback(() => {
    setConstraints((prev) => ({
      ...prev,
      facingMode: prev.facingMode === 'user' ? 'environment' : 'user',
    }));
  }, []);

  // Zoom controls
  const adjustZoom = useCallback((delta: number) => {
    setConstraints((prev) => ({
      ...prev,
      zoom: Math.max(1, Math.min(prev.zoom + delta, capabilities.maxZoom)),
    }));
  }, [capabilities.maxZoom]);

  // Capture image
  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0);

    // Get image data
    const imageData = canvas.toDataURL('image/jpeg', 0.95);
    setCapturedImage(imageData);
  }, []);

  // Use captured image
  const useImage = useCallback(async () => {
    if (!capturedImage) return;

    // Convert data URL to File
    const response = await fetch(capturedImage);
    const blob = await response.blob();
    const file = new File([blob], `camera-capture-${Date.now()}.jpg`, {
      type: 'image/jpeg',
    });

    onCapture(file);
    onClose();
  }, [capturedImage, onCapture, onClose]);

  // Retake photo
  const retakePhoto = useCallback(() => {
    setCapturedImage(null);
  }, []);

  // Render error state
  if (error && !isInitializing) {
    return (
      <div className="fixed inset-0 z-50 bg-background flex items-center justify-center p-4">
        <div className="max-w-md w-full space-y-6 text-center">
          <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center mx-auto">
            <AlertCircle className="w-8 h-8 text-destructive" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-foreground">Camera Error</h2>
            <p className="text-muted-foreground mt-2">{error}</p>
          </div>
          <div className="flex gap-3 justify-center">
            <button
              onClick={initializeCamera}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-border hover:bg-muted transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4 bg-gradient-to-b from-black/70 to-transparent">
        <div className="flex items-center justify-between">
          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center text-white hover:bg-white/30 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          <div className="text-white text-center">
            <p className="text-sm font-medium">Camera Capture</p>
            <p className="text-xs text-white/70">Position yourself in the frame</p>
          </div>
          <div className="w-10" /> {/* Spacer for alignment */}
        </div>
      </div>

      {/* Video/Image Display */}
      <div className="flex-1 relative">
        {isInitializing ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center space-y-4">
              <Loader2 className="w-10 h-10 text-white animate-spin mx-auto" />
              <p className="text-white">Initializing camera...</p>
            </div>
          </div>
        ) : capturedImage ? (
          <img
            src={capturedImage}
            alt="Captured"
            className="absolute inset-0 w-full h-full object-contain"
          />
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-contain"
            />

            {/* Pose Guide Overlay */}
            <PoseGuideOverlay />
          </>
        )}

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Controls */}
      <div className="absolute bottom-0 left-0 right-0 z-10 p-6 pb-safe bg-gradient-to-t from-black/70 to-transparent">
        {capturedImage ? (
          // Post-capture controls
          <div className="flex items-center justify-center gap-6">
            <button
              onClick={retakePhoto}
              className="flex flex-col items-center gap-2 text-white"
            >
              <div className="w-14 h-14 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center hover:bg-white/30 transition-colors">
                <RotateCcw className="w-6 h-6" />
              </div>
              <span className="text-xs">Retake</span>
            </button>

            <button
              onClick={useImage}
              className="flex flex-col items-center gap-2 text-white"
            >
              <div className="w-20 h-20 rounded-full bg-green-500 flex items-center justify-center hover:bg-green-600 transition-colors">
                <Check className="w-10 h-10" />
              </div>
              <span className="text-xs font-medium">Use Photo</span>
            </button>

            <div className="w-14" /> {/* Spacer */}
          </div>
        ) : (
          // Pre-capture controls
          <div className="flex items-center justify-center gap-6">
            {/* Zoom Out */}
            {capabilities.supportsZoom && (
              <button
                onClick={() => adjustZoom(-0.5)}
                disabled={constraints.zoom <= 1}
                className="flex flex-col items-center gap-2 text-white disabled:opacity-30"
              >
                <div className="w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center hover:bg-white/30 transition-colors">
                  <ZoomOut className="w-5 h-5" />
                </div>
              </button>
            )}

            {/* Capture Button */}
            <button
              onClick={captureImage}
              className="flex flex-col items-center gap-2 text-white"
            >
              <div className="w-20 h-20 rounded-full bg-white border-4 border-white/50 flex items-center justify-center hover:scale-95 transition-transform">
                <Camera className="w-8 h-8 text-black" />
              </div>
              <span className="text-xs">Capture</span>
            </button>

            {/* Zoom In */}
            {capabilities.supportsZoom && (
              <button
                onClick={() => adjustZoom(0.5)}
                disabled={constraints.zoom >= capabilities.maxZoom}
                className="flex flex-col items-center gap-2 text-white disabled:opacity-30"
              >
                <div className="w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center hover:bg-white/30 transition-colors">
                  <ZoomIn className="w-5 h-5" />
                </div>
              </button>
            )}
          </div>
        )}

        {/* Switch Camera Button */}
        {!capturedImage && (
          <div className="absolute right-6 bottom-1/2 transform translate-y-1/2">
            <button
              onClick={switchCamera}
              className="w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center text-white hover:bg-white/30 transition-colors"
            >
              <SwitchCamera className="w-5 h-5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// Pose Guide Overlay Component
function PoseGuideOverlay() {
  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg
        className="w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Body outline guide */}
        <g className="stroke-white/50 fill-none" strokeWidth="0.3" strokeDasharray="1,1">
          {/* Head */}
          <ellipse cx="50" cy="15" rx="6" ry="7" />

          {/* Neck */}
          <line x1="50" y1="22" x2="50" y2="26" />

          {/* Shoulders */}
          <line x1="35" y1="28" x2="65" y2="28" />

          {/* Body outline */}
          <path d="M 35 28 L 35 55 L 38 75 L 38 95" />
          <path d="M 65 28 L 65 55 L 62 75 L 62 95" />

          {/* Arms */}
          <path d="M 35 28 L 25 45 L 22 65" />
          <path d="M 65 28 L 75 45 L 78 65" />

          {/* Torso */}
          <line x1="35" y1="55" x2="65" y2="55" />
        </g>

        {/* Guide points */}
        <g className="fill-white/70">
          <circle cx="50" cy="15" r="1" /> {/* Head */}
          <circle cx="35" cy="28" r="1" /> {/* Left shoulder */}
          <circle cx="65" cy="28" r="1" /> {/* Right shoulder */}
          <circle cx="35" cy="55" r="1" /> {/* Left hip */}
          <circle cx="65" cy="55" r="1" /> {/* Right hip */}
        </g>
      </svg>

      {/* Instructions */}
      <div className="absolute bottom-32 left-0 right-0 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/50 backdrop-blur-sm text-white text-sm">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          Align your body with the guide
        </div>
      </div>

      {/* Corner guides */}
      <div className="absolute top-1/4 left-1/4 w-8 h-8 border-t-2 border-l-2 border-white/50 rounded-tl-lg" />
      <div className="absolute top-1/4 right-1/4 w-8 h-8 border-t-2 border-r-2 border-white/50 rounded-tr-lg" />
      <div className="absolute bottom-1/4 left-1/4 w-8 h-8 border-b-2 border-l-2 border-white/50 rounded-bl-lg" />
      <div className="absolute bottom-1/4 right-1/4 w-8 h-8 border-b-2 border-r-2 border-white/50 rounded-br-lg" />
    </div>
  );
}

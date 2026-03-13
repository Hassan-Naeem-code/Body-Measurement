'use client';

import { useState, useCallback, useRef } from 'react';
import { measurementsAPI } from '@/lib/api';
import { toast } from 'sonner';

interface DualViewUploadProps {
  apiKey: string;
  onResult?: (result: any) => void;
  heightCm?: number;
  productId?: string;
  fitPreference?: 'tight' | 'regular' | 'loose';
}

export function DualViewUpload({
  apiKey,
  onResult,
  heightCm,
  productId,
  fitPreference,
}: DualViewUploadProps) {
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [frontPreview, setFrontPreview] = useState<string | null>(null);
  const [sidePreview, setSidePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const frontRef = useRef<HTMLInputElement>(null);
  const sideRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File, type: 'front' | 'side') => {
      const url = URL.createObjectURL(file);
      if (type === 'front') {
        setFrontFile(file);
        setFrontPreview(url);
      } else {
        setSideFile(file);
        setSidePreview(url);
      }
    },
    []
  );

  const handleSubmit = useCallback(async () => {
    if (!frontFile || !sideFile) {
      toast.error('Please upload both a front and side photo.');
      return;
    }

    setIsProcessing(true);
    try {
      const result = await measurementsAPI.processDualView(
        apiKey,
        frontFile,
        sideFile,
        { heightCm, productId, fitPreference }
      );
      toast.success('Dual-view measurements processed successfully!');
      onResult?.(result);
    } catch {
      // Error toast handled by API interceptor
    } finally {
      setIsProcessing(false);
    }
  }, [frontFile, sideFile, apiKey, heightCm, productId, fitPreference, onResult]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Dual-View Upload (Higher Accuracy)</h3>
      <p className="text-sm text-gray-500">
        Upload both a front-facing and side-facing full-body photo for the most
        accurate measurements.
      </p>

      <div className="grid grid-cols-2 gap-4">
        {/* Front photo */}
        <div
          className="border-2 border-dashed rounded-lg p-4 text-center cursor-pointer hover:border-blue-400 transition-colors"
          onClick={() => frontRef.current?.click()}
        >
          <input
            ref={frontRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file, 'front');
            }}
          />
          {frontPreview ? (
            <img
              src={frontPreview}
              alt="Front view"
              className="mx-auto max-h-48 object-contain rounded"
            />
          ) : (
            <div className="py-8">
              <p className="font-medium">Front Photo</p>
              <p className="text-xs text-gray-400 mt-1">Face the camera</p>
            </div>
          )}
        </div>

        {/* Side photo */}
        <div
          className="border-2 border-dashed rounded-lg p-4 text-center cursor-pointer hover:border-blue-400 transition-colors"
          onClick={() => sideRef.current?.click()}
        >
          <input
            ref={sideRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file, 'side');
            }}
          />
          {sidePreview ? (
            <img
              src={sidePreview}
              alt="Side view"
              className="mx-auto max-h-48 object-contain rounded"
            />
          ) : (
            <div className="py-8">
              <p className="font-medium">Side Photo</p>
              <p className="text-xs text-gray-400 mt-1">Turn 90° to the side</p>
            </div>
          )}
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={!frontFile || !sideFile || isProcessing}
        className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
      >
        {isProcessing ? 'Processing...' : 'Process Dual-View Measurements'}
      </button>
    </div>
  );
}

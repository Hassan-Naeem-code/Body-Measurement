'use client';

import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';

// Dynamically import the 3D viewer with no SSR
// This is necessary because Three.js requires browser APIs
const Body3DViewerInner = dynamic(
  () => import('./body-3d-viewer-inner'),
  {
    ssr: false,
    loading: () => (
      <div className="relative bg-gray-900 rounded-lg overflow-hidden h-[500px] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-indigo-500 mx-auto mb-2" />
          <p className="text-gray-400">Loading 3D viewer...</p>
        </div>
      </div>
    ),
  }
);

type ViewMode = 'mannequin' | 'texture' | 'depth2.5d' | 'pifuhd';

interface Body3DViewerProps {
  heightCm?: number;
  gender?: 'male' | 'female' | 'neutral';
  weightFactor?: number;
  apiBaseUrl?: string;
  showControls?: boolean;
  showMeasurements?: boolean;
  autoRotate?: boolean;
  className?: string;
  imageUrl?: string; // URL of the uploaded image for texture mapping
  viewMode?: ViewMode; // Which 3D visualization mode to use
  onViewModeChange?: (mode: ViewMode) => void; // Callback when mode changes
}

export default function Body3DViewer(props: Body3DViewerProps) {
  return <Body3DViewerInner {...props} />;
}

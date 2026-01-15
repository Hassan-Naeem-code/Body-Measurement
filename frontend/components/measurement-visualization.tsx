'use client';

import { useRef, useEffect, useState } from 'react';
import type { PersonMeasurement, PoseLandmarks, PoseLandmark } from '@/lib/types';

interface MeasurementVisualizationProps {
  imageUrl: string;
  measurements: PersonMeasurement[];
  selectedPersonId?: number;
  showSkeleton?: boolean;
  showMeasurements?: boolean;
  showBoundingBox?: boolean;
}

// Colors for different people
const PERSON_COLORS = [
  { primary: '#6366f1', secondary: 'rgba(99, 102, 241, 0.3)' },
  { primary: '#10b981', secondary: 'rgba(16, 185, 129, 0.3)' },
  { primary: '#f59e0b', secondary: 'rgba(245, 158, 11, 0.3)' },
  { primary: '#ef4444', secondary: 'rgba(239, 68, 68, 0.3)' },
  { primary: '#8b5cf6', secondary: 'rgba(139, 92, 246, 0.3)' },
];

// Skeleton connections
const SKELETON_CONNECTIONS: [keyof PoseLandmarks, keyof PoseLandmarks][] = [
  ['nose', 'left_eye'],
  ['nose', 'right_eye'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
];

// Measurement lines
const MEASUREMENT_LINES: {
  name: string;
  from: keyof PoseLandmarks;
  to: keyof PoseLandmarks;
  measurementKey: keyof PersonMeasurement;
  unit: string;
  color: string;
  offset: number;
}[] = [
  {
    name: 'Shoulder',
    from: 'left_shoulder',
    to: 'right_shoulder',
    measurementKey: 'shoulder_width',
    unit: 'cm',
    color: '#f59e0b',
    offset: -30,
  },
  {
    name: 'Hip',
    from: 'left_hip',
    to: 'right_hip',
    measurementKey: 'hip_width',
    unit: 'cm',
    color: '#10b981',
    offset: 30,
  },
];

export function MeasurementVisualization({
  imageUrl,
  measurements,
  selectedPersonId,
  showSkeleton = true,
  showMeasurements = true,
  showBoundingBox = true,
}: MeasurementVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      setImageDimensions({ width: img.width, height: img.height });
      setImageLoaded(true);
    };
    img.src = imageUrl;

    return () => {
      imageRef.current = null;
    };
  }, [imageUrl]);

  // Calculate canvas size to fit container while maintaining aspect ratio
  useEffect(() => {
    if (!imageLoaded || !containerRef.current || !imageDimensions.width) {
      return;
    }

    const container = containerRef.current;
    const containerWidth = container.clientWidth;
    const maxHeight = window.innerHeight * 0.7; // 70vh max

    const imageAspect = imageDimensions.width / imageDimensions.height;

    let displayWidth = containerWidth;
    let displayHeight = containerWidth / imageAspect;

    // If height exceeds max, scale down
    if (displayHeight > maxHeight) {
      displayHeight = maxHeight;
      displayWidth = maxHeight * imageAspect;
    }

    setCanvasSize({ width: displayWidth, height: displayHeight });
  }, [imageLoaded, imageDimensions]);

  // Draw visualization
  useEffect(() => {
    if (!imageLoaded || !canvasRef.current || !imageRef.current || canvasSize.width === 0) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size (use device pixel ratio for crisp rendering)
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize.width * dpr;
    canvas.height = canvasSize.height * dpr;
    canvas.style.width = `${canvasSize.width}px`;
    canvas.style.height = `${canvasSize.height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

    // Draw image maintaining aspect ratio
    ctx.drawImage(imageRef.current, 0, 0, canvasSize.width, canvasSize.height);

    // Draw visualization for each person
    const peopleToShow = selectedPersonId !== undefined
      ? measurements.filter(m => m.person_id === selectedPersonId)
      : measurements;

    peopleToShow.forEach((person) => {
      const colorIndex = person.person_id % PERSON_COLORS.length;
      const colors = PERSON_COLORS[colorIndex];

      if (showBoundingBox && person.bounding_box) {
        drawBoundingBox(ctx, person.bounding_box, colors, canvasSize.width, canvasSize.height, person.person_id);
      }

      if (showSkeleton && person.pose_landmarks) {
        drawSkeleton(ctx, person.pose_landmarks, colors, canvasSize.width, canvasSize.height);
      }

      if (showMeasurements && person.pose_landmarks) {
        drawMeasurements(ctx, person.pose_landmarks, person, canvasSize.width, canvasSize.height);
      }
    });
  }, [imageLoaded, measurements, selectedPersonId, showSkeleton, showMeasurements, showBoundingBox, canvasSize]);

  return (
    <div ref={containerRef} className="relative w-full flex justify-center">
      <canvas
        ref={canvasRef}
        className="rounded-lg"
        style={{
          width: canvasSize.width || 'auto',
          height: canvasSize.height || 'auto',
          maxWidth: '100%',
        }}
      />
      {!imageLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
          <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
    </div>
  );
}

function drawBoundingBox(
  ctx: CanvasRenderingContext2D,
  bbox: { x1: number; y1: number; x2: number; y2: number },
  colors: { primary: string; secondary: string },
  canvasWidth: number,
  canvasHeight: number,
  personId: number
) {
  const x1 = bbox.x1 * canvasWidth;
  const y1 = bbox.y1 * canvasHeight;
  const x2 = bbox.x2 * canvasWidth;
  const y2 = bbox.y2 * canvasHeight;
  const width = x2 - x1;
  const height = y2 - y1;

  ctx.fillStyle = colors.secondary;
  ctx.fillRect(x1, y1, width, height);

  ctx.strokeStyle = colors.primary;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, width, height);

  const label = `Person ${personId + 1}`;
  ctx.font = 'bold 14px Inter, system-ui, sans-serif';
  const textWidth = ctx.measureText(label).width;

  ctx.fillStyle = colors.primary;
  ctx.fillRect(x1, y1 - 24, textWidth + 12, 22);

  ctx.fillStyle = 'white';
  ctx.fillText(label, x1 + 6, y1 - 8);
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  colors: { primary: string; secondary: string },
  canvasWidth: number,
  canvasHeight: number
) {
  ctx.strokeStyle = colors.primary;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';

  SKELETON_CONNECTIONS.forEach(([from, to]) => {
    const fromLandmark = landmarks[from];
    const toLandmark = landmarks[to];

    if (fromLandmark && toLandmark && fromLandmark.visibility > 0.3 && toLandmark.visibility > 0.3) {
      ctx.beginPath();
      ctx.moveTo(fromLandmark.x * canvasWidth, fromLandmark.y * canvasHeight);
      ctx.lineTo(toLandmark.x * canvasWidth, toLandmark.y * canvasHeight);
      ctx.stroke();
    }
  });

  Object.entries(landmarks).forEach(([name, landmark]) => {
    if (landmark && landmark.visibility > 0.3) {
      const x = landmark.x * canvasWidth;
      const y = landmark.y * canvasHeight;

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = colors.primary;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
    }
  });
}

function drawMeasurements(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  person: PersonMeasurement,
  canvasWidth: number,
  canvasHeight: number
) {
  MEASUREMENT_LINES.forEach((measurement) => {
    const fromLandmark = landmarks[measurement.from];
    const toLandmark = landmarks[measurement.to];
    const value = person[measurement.measurementKey] as number | null;

    if (!fromLandmark || !toLandmark || !value) return;
    if (fromLandmark.visibility < 0.3 || toLandmark.visibility < 0.3) return;

    const x1 = fromLandmark.x * canvasWidth;
    const y1 = fromLandmark.y * canvasHeight + measurement.offset;
    const x2 = toLandmark.x * canvasWidth;
    const y2 = toLandmark.y * canvasHeight + measurement.offset;
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;

    ctx.strokeStyle = measurement.color;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);

    const capLength = 10;
    ctx.beginPath();
    ctx.moveTo(x1, y1 - capLength / 2);
    ctx.lineTo(x1, y1 + capLength / 2);
    ctx.moveTo(x2, y2 - capLength / 2);
    ctx.lineTo(x2, y2 + capLength / 2);
    ctx.stroke();

    const label = `${measurement.name}: ${value.toFixed(1)}${measurement.unit}`;
    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = measurement.color;
    ctx.fillRect(midX - textWidth / 2 - 6, midY - 10, textWidth + 12, 20);

    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, midX, midY);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  });

  drawCircumferenceLabels(ctx, person, canvasWidth, canvasHeight);
}

function drawCircumferenceLabels(
  ctx: CanvasRenderingContext2D,
  person: PersonMeasurement,
  canvasWidth: number,
  canvasHeight: number
) {
  const circumferences = [
    { name: 'Chest', value: person.chest_circumference, color: '#6366f1' },
    { name: 'Waist', value: person.waist_circumference, color: '#ef4444' },
    { name: 'Hip', value: person.hip_circumference, color: '#10b981' },
  ];

  const startY = 30;
  const lineHeight = 28;
  const padding = 10;

  circumferences.forEach((circ, index) => {
    if (!circ.value) return;

    const y = startY + index * lineHeight;
    const label = `${circ.name}: ${circ.value.toFixed(1)}cm`;

    ctx.font = 'bold 13px Inter, system-ui, sans-serif';
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(padding, y - 9, textWidth + 16, 22);

    ctx.fillStyle = circ.color;
    ctx.fillRect(padding + 4, y - 5, 4, 14);

    ctx.fillStyle = 'white';
    ctx.fillText(label, padding + 12, y + 5);
  });
}

export function MeasurementVisualizationCompact({
  imageUrl,
  measurements,
}: {
  imageUrl: string;
  measurements: PersonMeasurement[];
}) {
  return (
    <MeasurementVisualization
      imageUrl={imageUrl}
      measurements={measurements}
      showSkeleton={true}
      showMeasurements={false}
      showBoundingBox={true}
    />
  );
}

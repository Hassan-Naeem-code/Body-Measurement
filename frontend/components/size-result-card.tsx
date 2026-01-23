'use client';

import { useState } from 'react';
import {
  ShoppingBag,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Info,
  ChevronDown,
  ChevronUp,
  Ruler,
  User,
  TrendingUp,
  Shield,
} from 'lucide-react';
import type { PersonMeasurement } from '@/lib/types';

interface SizeResultCardProps {
  person: PersonMeasurement;
  personIndex: number;
}

// Calculate overall confidence based on multiple factors
function calculateOverallConfidence(person: PersonMeasurement): {
  score: number;
  level: 'high' | 'medium' | 'low';
  factors: { name: string; status: 'good' | 'warning' | 'bad'; message: string }[];
} {
  const factors: { name: string; status: 'good' | 'warning' | 'bad'; message: string }[] = [];
  let totalScore = 0;
  let factorCount = 0;

  // Detection confidence
  if (person.detection_confidence >= 0.9) {
    factors.push({ name: 'Detection', status: 'good', message: 'Person clearly detected' });
    totalScore += 1;
  } else if (person.detection_confidence >= 0.7) {
    factors.push({ name: 'Detection', status: 'warning', message: 'Person partially visible' });
    totalScore += 0.6;
  } else {
    factors.push({ name: 'Detection', status: 'bad', message: 'Person hard to detect' });
    totalScore += 0.3;
  }
  factorCount++;

  // Validation confidence
  if (person.validation_confidence >= 0.85) {
    factors.push({ name: 'Body Visibility', status: 'good', message: 'Full body visible' });
    totalScore += 1;
  } else if (person.validation_confidence >= 0.6) {
    factors.push({ name: 'Body Visibility', status: 'warning', message: 'Some body parts unclear' });
    totalScore += 0.6;
  } else {
    factors.push({ name: 'Body Visibility', status: 'bad', message: 'Body partially hidden' });
    totalScore += 0.3;
  }
  factorCount++;

  // Pose angle (if available)
  if (person.pose_angle_degrees !== undefined && person.pose_angle_degrees !== null) {
    if (person.pose_angle_degrees <= 15) {
      factors.push({ name: 'Pose', status: 'good', message: 'Front-facing pose' });
      totalScore += 1;
    } else if (person.pose_angle_degrees <= 35) {
      factors.push({ name: 'Pose', status: 'warning', message: 'Slightly angled' });
      totalScore += 0.7;
    } else {
      factors.push({ name: 'Pose', status: 'bad', message: 'Sideways pose reduces accuracy' });
      totalScore += 0.4;
    }
    factorCount++;
  }

  // Body part confidences
  const bodyPartConfidences = person.body_part_confidences || {};
  const avgBodyPartConf = Object.values(bodyPartConfidences).length > 0
    ? Object.values(bodyPartConfidences).reduce((a, b) => a + b, 0) / Object.values(bodyPartConfidences).length
    : 0.7;

  if (avgBodyPartConf >= 0.8) {
    factors.push({ name: 'Landmarks', status: 'good', message: 'Body landmarks clear' });
    totalScore += 1;
  } else if (avgBodyPartConf >= 0.6) {
    factors.push({ name: 'Landmarks', status: 'warning', message: 'Some landmarks unclear' });
    totalScore += 0.6;
  } else {
    factors.push({ name: 'Landmarks', status: 'bad', message: 'Landmarks hard to detect' });
    totalScore += 0.3;
  }
  factorCount++;

  const score = totalScore / factorCount;
  const level = score >= 0.8 ? 'high' : score >= 0.6 ? 'medium' : 'low';

  return { score, level, factors };
}

// Get confidence color and label
function getConfidenceDisplay(level: 'high' | 'medium' | 'low') {
  switch (level) {
    case 'high':
      return {
        color: 'bg-green-500',
        bgColor: 'bg-green-50 dark:bg-green-900/20',
        textColor: 'text-green-700 dark:text-green-300',
        borderColor: 'border-green-200 dark:border-green-800',
        label: 'High Confidence',
        description: 'We\'re confident about this size recommendation',
        icon: CheckCircle2,
      };
    case 'medium':
      return {
        color: 'bg-yellow-500',
        bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
        textColor: 'text-yellow-700 dark:text-yellow-300',
        borderColor: 'border-yellow-200 dark:border-yellow-800',
        label: 'Medium Confidence',
        description: 'Size recommendation may vary by ±1 size',
        icon: AlertTriangle,
      };
    case 'low':
      return {
        color: 'bg-red-500',
        bgColor: 'bg-red-50 dark:bg-red-900/20',
        textColor: 'text-red-700 dark:text-red-300',
        borderColor: 'border-red-200 dark:border-red-800',
        label: 'Low Confidence',
        description: 'Photo quality affects accuracy - consider a clearer photo',
        icon: AlertCircle,
      };
  }
}

export function SizeResultCard({ person, personIndex }: SizeResultCardProps) {
  const [showDetails, setShowDetails] = useState(false);
  const confidence = calculateOverallConfidence(person);
  const confidenceDisplay = getConfidenceDisplay(confidence.level);
  const ConfidenceIcon = confidenceDisplay.icon;

  // If person is not valid, show different card
  if (!person.is_valid) {
    return (
      <div className="rounded-2xl border-2 border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 overflow-hidden">
        <div className="p-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/40 flex items-center justify-center flex-shrink-0">
              <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
                Person {personIndex + 1} - Unable to Measure
              </h3>
              <p className="text-red-600 dark:text-red-300 mt-1">
                We couldn't get accurate measurements for this person.
              </p>

              {person.missing_parts && person.missing_parts.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-2">Issues detected:</p>
                  <ul className="space-y-1">
                    {person.missing_parts.map((part, idx) => (
                      <li key={idx} className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                        {part.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mt-4 p-3 rounded-lg bg-red-100 dark:bg-red-900/40">
                <p className="text-sm font-medium text-red-800 dark:text-red-200 mb-1">Tips for better results:</p>
                <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
                  <li>• Use a photo showing the full body (head to feet)</li>
                  <li>• Person should be facing the camera</li>
                  <li>• Avoid photos where person is cut off or obstructed</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border-2 border-border bg-card overflow-hidden shadow-lg">
      {/* Main Result - The Size */}
      <div className="p-6 bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 text-white/80 text-sm">
              <User className="w-4 h-4" />
              <span>Person {personIndex + 1}</span>
              {person.demographic_label && (
                <>
                  <span className="text-white/50">•</span>
                  <span>{person.demographic_label}</span>
                </>
              )}
            </div>
            <div className="mt-3">
              <p className="text-white/80 text-sm font-medium uppercase tracking-wide">
                Recommended Size
              </p>
              <div className="flex items-baseline gap-3 mt-1">
                <span className="text-6xl font-bold">{person.recommended_size || 'M'}</span>
                {person.size_probabilities && (
                  <span className="text-white/70 text-lg">
                    ({Math.round((person.size_probabilities[person.recommended_size || 'M'] || 0) * 100)}% match)
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="text-right">
            <ShoppingBag className="w-16 h-16 text-white/20" />
          </div>
        </div>
      </div>

      {/* Confidence Indicator */}
      <div className={`p-4 ${confidenceDisplay.bgColor} border-b ${confidenceDisplay.borderColor}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full ${confidenceDisplay.bgColor} flex items-center justify-center`}>
              <ConfidenceIcon className={`w-5 h-5 ${confidenceDisplay.textColor}`} />
            </div>
            <div>
              <p className={`font-semibold ${confidenceDisplay.textColor}`}>
                {confidenceDisplay.label}
              </p>
              <p className="text-sm text-muted-foreground">
                {confidenceDisplay.description}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Confidence bar */}
            <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full ${confidenceDisplay.color} rounded-full transition-all`}
                style={{ width: `${confidence.score * 100}%` }}
              />
            </div>
            <span className={`text-sm font-medium ${confidenceDisplay.textColor}`}>
              {Math.round(confidence.score * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* Alternative Sizes */}
      {person.size_probabilities && Object.keys(person.size_probabilities).length > 1 && (
        <div className="p-4 border-b border-border">
          <p className="text-sm font-medium text-muted-foreground mb-3">Size Options</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(person.size_probabilities)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 4)
              .map(([size, probability]) => (
                <div
                  key={size}
                  className={`px-4 py-2 rounded-lg border-2 ${
                    size === person.recommended_size
                      ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30'
                      : 'border-border bg-muted/30'
                  }`}
                >
                  <span className={`font-bold ${size === person.recommended_size ? 'text-indigo-600 dark:text-indigo-400' : 'text-foreground'}`}>
                    {size}
                  </span>
                  <span className="text-sm text-muted-foreground ml-2">
                    {Math.round(probability * 100)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Expandable Details */}
      <div className="border-t border-border">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full p-4 flex items-center justify-between text-sm text-muted-foreground hover:bg-muted/30 transition-colors"
        >
          <span className="flex items-center gap-2">
            <Info className="w-4 h-4" />
            {showDetails ? 'Hide Details' : 'Show Measurement Details'}
          </span>
          {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showDetails && (
          <div className="px-4 pb-4 space-y-4">
            {/* Measurements Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {person.chest_circumference && (
                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-xs text-muted-foreground">Chest</p>
                  <p className="text-lg font-semibold">{person.chest_circumference.toFixed(1)} cm</p>
                </div>
              )}
              {person.waist_circumference && (
                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-xs text-muted-foreground">Waist</p>
                  <p className="text-lg font-semibold">{person.waist_circumference.toFixed(1)} cm</p>
                </div>
              )}
              {person.hip_circumference && (
                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-xs text-muted-foreground">Hip</p>
                  <p className="text-lg font-semibold">{person.hip_circumference.toFixed(1)} cm</p>
                </div>
              )}
              {person.shoulder_width && (
                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-xs text-muted-foreground">Shoulder Width</p>
                  <p className="text-lg font-semibold">{person.shoulder_width.toFixed(1)} cm</p>
                </div>
              )}
              {person.inseam && (
                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-xs text-muted-foreground">Inseam</p>
                  <p className="text-lg font-semibold">{person.inseam.toFixed(1)} cm</p>
                </div>
              )}
              {person.estimated_height_cm && (
                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-xs text-muted-foreground">Est. Height</p>
                  <p className="text-lg font-semibold">{person.estimated_height_cm.toFixed(0)} cm</p>
                </div>
              )}
            </div>

            {/* Confidence Factors */}
            <div>
              <p className="text-sm font-medium text-foreground mb-2 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Confidence Factors
              </p>
              <div className="space-y-2">
                {confidence.factors.map((factor, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{factor.name}</span>
                    <span className={`flex items-center gap-1 ${
                      factor.status === 'good' ? 'text-green-600 dark:text-green-400' :
                      factor.status === 'warning' ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-red-600 dark:text-red-400'
                    }`}>
                      {factor.status === 'good' && <CheckCircle2 className="w-3 h-3" />}
                      {factor.status === 'warning' && <AlertTriangle className="w-3 h-3" />}
                      {factor.status === 'bad' && <AlertCircle className="w-3 h-3" />}
                      {factor.message}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

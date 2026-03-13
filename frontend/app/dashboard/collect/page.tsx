'use client';

import { useState, useRef, useCallback } from 'react';
import { authHelpers } from '@/lib/auth';
import { groundTruthAPI } from '@/lib/api';
import { toast } from 'sonner';
import {
  Upload,
  Camera,
  Ruler,
  CheckCircle2,
  Info,
  ChevronDown,
  ChevronUp,
  Trash2,
} from 'lucide-react';

interface MeasurementField {
  key: string;
  label: string;
  placeholder: string;
  hint: string;
  unit: string;
  min: number;
  max: number;
}

const MEASUREMENT_FIELDS: MeasurementField[] = [
  { key: 'actual_height', label: 'Height', placeholder: '170', hint: 'Stand straight against a wall, measure from floor to top of head', unit: 'cm', min: 100, max: 220 },
  { key: 'actual_chest_circumference', label: 'Chest', placeholder: '96', hint: 'Wrap tape around the fullest part of your chest, under armpits', unit: 'cm', min: 60, max: 160 },
  { key: 'actual_waist_circumference', label: 'Waist', placeholder: '80', hint: 'Measure at your natural waistline (narrowest part of torso)', unit: 'cm', min: 50, max: 150 },
  { key: 'actual_hip_circumference', label: 'Hips', placeholder: '98', hint: 'Measure around the widest part of your hips/buttocks', unit: 'cm', min: 60, max: 160 },
  { key: 'actual_shoulder_width', label: 'Shoulder Width', placeholder: '45', hint: 'Measure from one shoulder edge to the other across the back', unit: 'cm', min: 25, max: 70 },
  { key: 'actual_inseam', label: 'Inseam', placeholder: '78', hint: 'Measure from crotch to ankle bone along the inner leg', unit: 'cm', min: 50, max: 110 },
  { key: 'actual_arm_length', label: 'Arm Length', placeholder: '60', hint: 'From shoulder point to wrist bone with arm slightly bent', unit: 'cm', min: 40, max: 90 },
];

export default function CollectDataPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<Record<string, string>>({});
  const [gender, setGender] = useState('');
  const [measuredBy, setMeasuredBy] = useState('');
  const [notes, setNotes] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [submitCount, setSubmitCount] = useState(0);
  const [showGuide, setShowGuide] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(selected.type)) {
      toast.error('Please upload a JPG, PNG, or WEBP image.');
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setSubmitted(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (!dropped) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(dropped.type)) {
      toast.error('Please upload a JPG, PNG, or WEBP image.');
      return;
    }
    setFile(dropped);
    setPreview(URL.createObjectURL(dropped));
    setSubmitted(false);
  }, []);

  const handleMeasurementChange = (key: string, value: string) => {
    // Allow empty or valid numbers only
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
      setMeasurements(prev => ({ ...prev, [key]: value }));
    }
  };

  const filledCount = MEASUREMENT_FIELDS.filter(f => measurements[f.key] && parseFloat(measurements[f.key]) > 0).length;

  const handleSubmit = async () => {
    if (!file) {
      toast.error('Please upload a full-body photo first.');
      return;
    }
    if (filledCount < 2) {
      toast.error('Please fill in at least 2 measurements.');
      return;
    }

    const apiKey = authHelpers.getApiKey();
    if (!apiKey) {
      toast.error('API key not found. Please check your API Keys page.');
      return;
    }

    setSubmitting(true);
    try {
      const data: Record<string, any> = {};
      MEASUREMENT_FIELDS.forEach(f => {
        const val = parseFloat(measurements[f.key]);
        if (!isNaN(val) && val > 0) {
          // Validate range
          if (val < f.min || val > f.max) {
            throw new Error(`${f.label} should be between ${f.min} and ${f.max} cm`);
          }
          data[f.key] = val;
        }
      });
      if (gender) data.gender = gender;
      if (measuredBy) data.measured_by = measuredBy;
      if (notes) data.notes = notes;

      await groundTruthAPI.submit(apiKey, file, data);
      setSubmitted(true);
      setSubmitCount(prev => prev + 1);
      toast.success('Sample submitted! Thank you for contributing.');

      // Reset form for next submission
      setFile(null);
      setPreview(null);
      setMeasurements({});
      setNotes('');
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      toast.error(err.message || 'Failed to submit. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setMeasurements({});
    setGender('');
    setMeasuredBy('');
    setNotes('');
    setSubmitted(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Collect Ground Truth Data</h1>
        <p className="mt-1 text-gray-500">
          Upload a full-body photo and enter your real tape measurements. This data trains our AI to be more accurate.
        </p>
        {submitCount > 0 && (
          <div className="mt-2 inline-flex items-center gap-2 px-3 py-1.5 bg-green-50 text-green-700 rounded-full text-sm font-medium">
            <CheckCircle2 className="w-4 h-4" />
            {submitCount} sample{submitCount !== 1 ? 's' : ''} submitted this session
          </div>
        )}
      </div>

      {/* How-to Guide (collapsible) */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowGuide(!showGuide)}
          className="w-full flex items-center justify-between p-4 text-left"
        >
          <div className="flex items-center gap-2 text-blue-800 font-medium">
            <Info className="w-5 h-5" />
            How to take measurements
          </div>
          {showGuide ? <ChevronUp className="w-5 h-5 text-blue-600" /> : <ChevronDown className="w-5 h-5 text-blue-600" />}
        </button>
        {showGuide && (
          <div className="px-4 pb-4 space-y-3 text-sm text-blue-800">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="bg-white/60 rounded-lg p-3">
                <p className="font-semibold mb-1">Photo Tips</p>
                <ul className="space-y-1 text-blue-700">
                  <li>- Stand straight, arms slightly away from body</li>
                  <li>- Full body visible from head to feet</li>
                  <li>- Wear fitted clothing (not baggy)</li>
                  <li>- Good lighting, plain background</li>
                  <li>- Camera at waist height, 2-3 meters away</li>
                </ul>
              </div>
              <div className="bg-white/60 rounded-lg p-3">
                <p className="font-semibold mb-1">Measurement Tips</p>
                <ul className="space-y-1 text-blue-700">
                  <li>- Use a flexible tape measure</li>
                  <li>- Keep tape snug but not tight</li>
                  <li>- Measure in cm (not inches)</li>
                  <li>- Ask a friend to help for accuracy</li>
                  <li>- You don't need ALL measurements — even 2-3 help!</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Photo Upload */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Camera className="w-5 h-5 text-indigo-600" />
          Step 1: Upload Full-Body Photo
        </h2>

        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="max-h-80 mx-auto rounded-lg object-contain border border-gray-200"
            />
            <button
              onClick={() => {
                setFile(null);
                setPreview(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              className="absolute top-2 right-2 p-2 bg-white/90 rounded-full shadow hover:bg-red-50 transition-colors"
            >
              <Trash2 className="w-4 h-4 text-red-500" />
            </button>
          </div>
        ) : (
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50/30 transition-colors"
          >
            <Upload className="w-10 h-10 mx-auto text-gray-400 mb-3" />
            <p className="text-gray-600 font-medium">Click or drag & drop your photo</p>
            <p className="text-sm text-gray-400 mt-1">JPG, PNG, or WEBP</p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Measurements */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-1 flex items-center gap-2">
          <Ruler className="w-5 h-5 text-indigo-600" />
          Step 2: Enter Your Tape Measurements
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Fill in as many as you can. At least 2 required. ({filledCount}/{MEASUREMENT_FIELDS.length} filled)
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {MEASUREMENT_FIELDS.map((field) => (
            <div key={field.key}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {field.label} <span className="text-gray-400">({field.unit})</span>
              </label>
              <input
                type="text"
                inputMode="decimal"
                value={measurements[field.key] || ''}
                onChange={(e) => handleMeasurementChange(field.key, e.target.value)}
                placeholder={field.placeholder}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-colors"
              />
              <p className="mt-0.5 text-xs text-gray-400">{field.hint}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Optional Info */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Optional Details</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
            <select
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
            >
              <option value="">-- Select --</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other / Prefer not to say</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Your Name (optional)</label>
            <input
              type="text"
              value={measuredBy}
              onChange={(e) => setMeasuredBy(e.target.value)}
              placeholder="e.g., Hassan"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
            />
          </div>
          <div className="sm:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g., Measured after dinner, wearing socks..."
              rows={2}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none resize-none"
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between gap-4">
        <button
          onClick={handleClear}
          className="px-4 py-2.5 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
        >
          Clear All
        </button>
        <button
          onClick={handleSubmit}
          disabled={submitting || !file || filledCount < 2}
          className="px-6 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {submitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Submitting...
            </>
          ) : (
            <>
              <CheckCircle2 className="w-4 h-4" />
              Submit Sample
            </>
          )}
        </button>
      </div>

      {/* Progress toward goal */}
      <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 text-center text-sm text-gray-500">
        <p className="font-medium text-gray-700 mb-1">Goal: 50+ samples for model training</p>
        <p>
          Ask friends, family, or colleagues to stand for a quick photo and tape-measure themselves.
          Each person takes ~3 minutes. Share this page link with them!
        </p>
      </div>
    </div>
  );
}

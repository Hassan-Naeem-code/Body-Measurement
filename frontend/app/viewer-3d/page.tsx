'use client';

import { useState } from 'react';
import Body3DViewer from '@/components/body-3d-viewer';
import { User, Ruler, Scale } from 'lucide-react';

export default function Viewer3DPage() {
  const [height, setHeight] = useState(175);
  const [gender, setGender] = useState<'male' | 'female' | 'neutral'>('neutral');
  const [weightFactor, setWeightFactor] = useState(0);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold text-indigo-400">3D Body Viewer</h1>
        <p className="text-gray-400 text-sm mt-1">
          Interactive 360° view of your body measurements
        </p>
      </header>

      <div className="flex flex-col lg:flex-row">
        {/* Controls Panel */}
        <div className="lg:w-80 p-6 border-r border-gray-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <User className="w-5 h-5 text-indigo-400" />
            Body Parameters
          </h2>

          {/* Height */}
          <div className="mb-6">
            <label className="block text-sm text-gray-400 mb-2 flex items-center gap-2">
              <Ruler className="w-4 h-4" />
              Height: {height} cm
            </label>
            <input
              type="range"
              min="140"
              max="210"
              value={height}
              onChange={(e) => setHeight(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>140 cm</span>
              <span>210 cm</span>
            </div>
          </div>

          {/* Gender */}
          <div className="mb-6">
            <label className="block text-sm text-gray-400 mb-2">Gender</label>
            <div className="grid grid-cols-3 gap-2">
              {(['male', 'female', 'neutral'] as const).map((g) => (
                <button
                  key={g}
                  onClick={() => setGender(g)}
                  className={`px-3 py-2 rounded-lg text-sm capitalize transition-colors ${
                    gender === g
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {g}
                </button>
              ))}
            </div>
          </div>

          {/* Weight Factor */}
          <div className="mb-6">
            <label className="block text-sm text-gray-400 mb-2 flex items-center gap-2">
              <Scale className="w-4 h-4" />
              Body Build: {weightFactor > 0 ? `+${weightFactor}` : weightFactor}
            </label>
            <input
              type="range"
              min="-3"
              max="3"
              step="0.5"
              value={weightFactor}
              onChange={(e) => setWeightFactor(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Thin</span>
              <span>Average</span>
              <span>Heavy</span>
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-8 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2">Controls</h3>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Drag to rotate the model</li>
              <li>• Scroll to zoom in/out</li>
              <li>• Right-click drag to pan</li>
              <li>• Click the rotate button to auto-spin</li>
            </ul>
          </div>

          {/* Color Legend */}
          <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2">Measurement Regions</h3>
            <ul className="text-xs space-y-2">
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 bg-red-500 rounded" />
                <span className="text-gray-400">Chest circumference</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 bg-green-500 rounded" />
                <span className="text-gray-400">Waist circumference</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 bg-blue-500 rounded" />
                <span className="text-gray-400">Hip circumference</span>
              </li>
            </ul>
          </div>
        </div>

        {/* 3D Viewer */}
        <div className="flex-1 p-6">
          <Body3DViewer
            heightCm={height}
            gender={gender}
            weightFactor={weightFactor}
            showControls={true}
            showMeasurements={true}
            autoRotate={true}
            className="h-[600px]"
          />
        </div>
      </div>
    </div>
  );
}

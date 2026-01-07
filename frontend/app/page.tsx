import Link from 'next/link';
import { Camera, Bot, Shirt, ArrowRight, BookOpen } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="px-6 py-24">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-3xl">
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-gray-900">
              Precision Sizing. Fewer Returns.
            </h1>
            <p className="mt-6 text-lg text-gray-600">
              AI-powered body measurements and size recommendations that boost conversions and
              reduce returns — integrate in minutes with a clean, modern API.
            </p>
            <div className="mt-10 flex gap-4">
              <Link
                href="/auth/register"
                className="inline-flex items-center gap-2 rounded-lg px-6 py-3 font-semibold shadow-sm transition
                           bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90"
              >
                Start Free
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 rounded-lg px-6 py-3 font-semibold transition
                           bg-white text-gray-900 border border-gray-200 hover:bg-gray-50"
              >
                See Dashboard
              </Link>
              <Link
                href="/auth/login"
                className="inline-flex items-center gap-2 rounded-lg px-6 py-3 font-semibold transition
                           bg-white text-gray-900 border border-gray-200 hover:bg-gray-50"
              >
                Sign In
              </Link>
            </div>
          </div>

          {/* Feature highlights */}
          <div className="mt-20 grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="w-10 h-10 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center mb-4">
                <Camera className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">Upload Photo</h3>
              <p className="mt-2 text-gray-600">Customers upload a single full-body photo.</p>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="w-10 h-10 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center mb-4">
                <Bot className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">AI Processing</h3>
              <p className="mt-2 text-gray-600">We extract precise measurements instantly.</p>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="w-10 h-10 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center mb-4">
                <Shirt className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">Perfect Fit</h3>
              <p className="mt-2 text-gray-600">Accurate, confidence-scored recommendations.</p>
            </div>
          </div>

          {/* CTA strip */}
          <div className="mt-24 rounded-lg border border-gray-200 p-6 bg-white flex items-center justify-between">
            <div>
              <h4 className="text-lg font-semibold text-gray-900">Integrate in minutes</h4>
              <p className="text-gray-600 mt-1">Clean REST API with simple, well-documented endpoints.</p>
            </div>
            <Link
              href="/dashboard/api-keys"
              className="inline-flex items-center gap-2 rounded-lg px-5 py-2.5 font-semibold shadow-sm transition
                         bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90"
            >
              View API Keys
              <BookOpen className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

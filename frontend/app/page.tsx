import Link from 'next/link';
import { Camera, Bot, Shirt, ArrowRight, BookOpen, CheckCircle2, Zap, Shield } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="border-b border-gray-100 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Shirt className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-gray-900">FitWhisperer</span>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/auth/login"
              className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition"
            >
              Sign In
            </Link>
            <Link
              href="/auth/register"
              className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="px-6 py-20 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-sm font-medium mb-6">
              <Zap className="w-4 h-4" />
              AI-Powered Sizing Technology
            </div>
            <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-gray-900 leading-tight">
              Precision Sizing.<br />Fewer Returns.
            </h1>
            <p className="mt-6 text-lg text-gray-600 leading-relaxed">
              AI-powered body measurements and size recommendations that boost conversions and
              reduce returns. Integrate in minutes with our clean, modern API.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link
                href="/auth/register"
                className="inline-flex items-center gap-2 px-6 py-3 text-base font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition shadow-sm"
              >
                Start Free
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 px-6 py-3 text-base font-semibold text-gray-700 bg-white rounded-lg border border-gray-200 hover:bg-gray-50 transition"
              >
                See Dashboard
              </Link>
            </div>
            <div className="mt-8 flex items-center gap-6 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                No credit card required
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                Free tier available
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="px-6 py-16">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-2xl font-bold text-gray-900 text-center mb-12">How it works</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center mb-4">
                <Camera className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">1. Upload Photo</h3>
              <p className="mt-2 text-gray-600">Customers upload a single full-body photo from their phone or computer.</p>
            </div>
            <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center mb-4">
                <Bot className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">2. AI Processing</h3>
              <p className="mt-2 text-gray-600">Our AI extracts precise body measurements in under 2 seconds.</p>
            </div>
            <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center mb-4">
                <Shirt className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">3. Perfect Fit</h3>
              <p className="mt-2 text-gray-600">Get accurate, confidence-scored size recommendations instantly.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="px-6 py-16 bg-gray-50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-2xl font-bold text-gray-900 text-center mb-4">Built for developers</h2>
          <p className="text-gray-600 text-center mb-12 max-w-2xl mx-auto">
            Simple REST API with comprehensive documentation. Get started in minutes.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <div className="w-10 h-10 rounded-lg bg-green-100 text-green-600 flex items-center justify-center mb-4">
                <Zap className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-gray-900">Fast Integration</h3>
              <p className="mt-2 text-sm text-gray-600">Simple API calls. Production ready in hours, not weeks.</p>
            </div>
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <div className="w-10 h-10 rounded-lg bg-blue-100 text-blue-600 flex items-center justify-center mb-4">
                <Shield className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-gray-900">Enterprise Security</h3>
              <p className="mt-2 text-sm text-gray-600">SOC2 compliant. Images processed and deleted immediately.</p>
            </div>
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <div className="w-10 h-10 rounded-lg bg-purple-100 text-purple-600 flex items-center justify-center mb-4">
                <BookOpen className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-gray-900">Great Documentation</h3>
              <p className="mt-2 text-sm text-gray-600">Comprehensive docs with code examples in every language.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="bg-indigo-600 rounded-2xl p-8 sm:p-12 text-center">
            <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4">Ready to reduce returns?</h2>
            <p className="text-indigo-100 mb-8 max-w-xl mx-auto">
              Join hundreds of e-commerce brands using our AI sizing technology.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/auth/register"
                className="inline-flex items-center gap-2 px-6 py-3 text-base font-semibold text-indigo-600 bg-white rounded-lg hover:bg-gray-50 transition"
              >
                Get Started Free
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                href="/dashboard/api-keys"
                className="inline-flex items-center gap-2 px-6 py-3 text-base font-semibold text-white border-2 border-white/30 rounded-lg hover:bg-white/10 transition"
              >
                View API Docs
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 border-t border-gray-100">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-indigo-600 flex items-center justify-center">
              <Shirt className="w-4 h-4 text-white" />
            </div>
            <span>FitWhisperer</span>
          </div>
          <p>Built for modern e-commerce</p>
        </div>
      </footer>
    </div>
  );
}

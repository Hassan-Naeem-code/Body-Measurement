import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Body Measurement API
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            AI-Powered Body Measurements & Size Recommendations for E-commerce
          </p>
          <p className="text-lg text-gray-500 mb-12">
            Reduce returns, increase conversions, and improve customer satisfaction with accurate size recommendations powered by computer vision.
          </p>

          <div className="flex gap-4 justify-center mb-16">
            <Link
              href="/auth/register"
              className="bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
            >
              Get Started
            </Link>
            <Link
              href="/auth/login"
              className="bg-white text-indigo-600 px-8 py-3 rounded-lg font-semibold border-2 border-indigo-600 hover:bg-indigo-50 transition"
            >
              Sign In
            </Link>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-4xl mb-4">📸</div>
              <h3 className="text-xl font-semibold mb-2">Upload Photo</h3>
              <p className="text-gray-600">
                Customers upload a single full-body photo
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-xl font-semibold mb-2">AI Processing</h3>
              <p className="text-gray-600">
                Our AI extracts precise body measurements
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-4xl mb-4">👔</div>
              <h3 className="text-xl font-semibold mb-2">Perfect Fit</h3>
              <p className="text-gray-600">
                Get accurate size recommendations instantly
              </p>
            </div>
          </div>

          <div className="mt-16 bg-white p-8 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Key Features</h2>
            <ul className="text-left max-w-2xl mx-auto space-y-3 text-gray-700">
              <li>✓ Real-time body measurement extraction</li>
              <li>✓ Personalized size recommendations</li>
              <li>✓ 95%+ accuracy with confidence scores</li>
              <li>✓ RESTful API for easy integration</li>
              <li>✓ Analytics & ROI tracking dashboard</li>
              <li>✓ Custom size charts per product</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

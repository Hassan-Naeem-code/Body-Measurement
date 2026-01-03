# Quick Start Guide

Get the Body Measurement API frontend up and running in 5 minutes!

## Step 1: Start the Development Server

```bash
cd ~/Desktop/body-measurement-frontend
npm run dev
```

The app will be available at **http://localhost:3000**

## Step 2: Ensure Backend is Running

Make sure your FastAPI backend is running on **http://localhost:8000**

If not, start it with:
```bash
cd /path/to/backend
docker-compose up
```

Verify backend is running: http://localhost:8000/docs

## Step 3: Create Your First Account

1. Open http://localhost:3000 in your browser
2. Click "Get Started" or "Sign In"
3. Click "Create one" to register
4. Fill in:
   - Brand Name: `Test Brand`
   - Email: `test@example.com`
   - Password: `testpassword123`
5. Click "Create Account"

You'll be automatically logged in and redirected to the dashboard!

## Step 4: Test the Features

### Upload an Image
1. Click "Upload Image" in the sidebar
2. Choose a full-body photo
3. Click "Process Measurements"
4. View detailed measurements and size recommendations

### Add a Product
1. Click "Products" in the sidebar
2. Click "+ Add Product"
3. Enter product details:
   - Name: `Classic T-Shirt`
   - Category: `T-Shirts`
   - Fill in size chart measurements
4. Click "Add Product"

### View Analytics
1. Click "Analytics" in the sidebar
2. See statistics, size distribution, and ROI metrics

### Get Your API Key
1. Click "API Keys" in the sidebar
2. Copy your API key
3. Use it to integrate with your platform

## Project Structure

```
body-measurement-frontend/
├── app/                    # Next.js App Router pages
│   ├── auth/              # Login & Register
│   ├── dashboard/         # Dashboard pages
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Landing page
├── lib/                   # Utilities
│   ├── api.ts            # API functions
│   ├── auth.ts           # Auth helpers
│   └── types.ts          # TypeScript types
├── .env.local            # Environment variables
└── package.json          # Dependencies
```

## Available Pages

| Page | URL | Description |
|------|-----|-------------|
| Landing | http://localhost:3000 | Home page |
| Register | http://localhost:3000/auth/register | Create account |
| Login | http://localhost:3000/auth/login | Sign in |
| Dashboard | http://localhost:3000/dashboard | Stats overview |
| Upload | http://localhost:3000/dashboard/upload | Process images |
| Products | http://localhost:3000/dashboard/products | Manage products |
| Analytics | http://localhost:3000/dashboard/analytics | View metrics |
| API Keys | http://localhost:3000/dashboard/api-keys | Get API key |

## Common Issues

### CORS Errors
**Problem**: Can't connect to backend

**Solution**: Add to backend `.env`:
```env
CORS_ORIGINS=http://localhost:3000
```
Then restart backend.

### "API key not found"
**Problem**: Lost authentication

**Solution**:
1. Clear browser localStorage
2. Log out and log back in

### Build Errors
**Problem**: npm run build fails

**Solution**:
```bash
rm -rf .next node_modules
npm install
npm run build
```

## Testing the API

Use cURL to test backend directly:

### Register
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Brand",
    "email": "test@brand.com",
    "password": "testpass123"
  }'
```

### Process Image
```bash
curl -X POST "http://localhost:8000/api/v1/measurements/process?api_key=YOUR_KEY" \
  -F "file=@image.jpg"
```

## Environment Variables

The `.env.local` file contains:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Change this if your backend runs on a different URL.

## Development Tips

### Hot Reload
Next.js automatically reloads when you save files

### TypeScript
All types are defined in `lib/types.ts`

### Styling
Uses Tailwind CSS - modify `tailwind.config.ts` for custom colors

### API Calls
All API functions are in `lib/api.ts` - add new endpoints there

## Production Build

When ready for production:

```bash
npm run build
npm start
```

This creates an optimized production build.

## Next Steps

1. ✅ Register and login
2. ✅ Upload a test image
3. ✅ Add a product with size chart
4. ✅ Check analytics
5. ✅ Get your API key
6. 🚀 Integrate with your e-commerce platform

## Need Help?

- **Backend API Docs**: http://localhost:8000/docs
- **Frontend Issues**: Check browser console (F12)
- **Backend Issues**: Check Docker logs

Happy coding! 🎉
